# Modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/s2_train.py

import os
import json
from random import randint

import torch.distributed as dist
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .module import commons
from .module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .module.losses import feature_loss, generator_loss, discriminator_loss, kl_loss
from .module.models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from .module.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from .utils import (
    get_hparams,
    get_logger,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    plot_spectrogram_to_numpy,
    summarize,
    savee,
    clear_gpu_cache,
)
from .config import TTSModel

# Prevent module not found error when loading pretrained models
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision('medium')


class S2Train:

    def __init__(self, batch_size, total_epoch, text_low_lr_rate,
                 if_save_latest, if_save_every_weights, save_every_epoch):
        self.model = TTSModel()

        self.global_step = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            os.makedirs(self.model.tmp_dir, exist_ok=True)
            os.makedirs(self.model.s2_dir, exist_ok=True)
            os.makedirs(self.model.s2_ckpt_dir, exist_ok=True)
            os.makedirs(self.model.sovits_weights_path, exist_ok=True)

            with open(self.model.s2_config_path, 'r') as f:
                data = f.read()
                data = json.loads(data)

            data['train']['fp16_run'] = False  # 1 gpu
            data['train']['batch_size'] = max(1, batch_size // 2)  # 1 gpu
            data['train']['epochs'] = total_epoch
            data['train']['text_low_lr_rate'] = text_low_lr_rate
            data['train']['pretrained_s2G'] = self.model.pretrained_s2G_path
            data['train']['pretrained_s2D'] = self.model.pretrained_s2D_path
            data['train']['if_save_latest'] = if_save_latest
            data['train']['if_save_every_weights'] = if_save_every_weights
            data['train']['save_every_epoch'] = save_every_epoch
            data['data']['exp_dir'] = self.model.s2_dir
            data['s2_ckpt_dir'] = self.model.s2_ckpt_dir
            data['save_weight_dir'] = self.model.sovits_weights_path

            self.tmp_config_path = os.path.join(
                self.model.tmp_dir, 'tmp_s2.json')
            with open(self.tmp_config_path, 'w') as f:
                f.write(json.dumps(data))
        except Exception as e:
            raise Exception(f'Error creating temporary config file: {e}')

        try:
            self.hps = get_hparams(
                config_path=self.tmp_config_path, stage=2)
        except Exception as e:
            raise Exception(
                f'Error getting hyperparameters from temporary config file: {e}')

    def run(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(randint(10000, 20000))

        logger = get_logger(self.hps.data.exp_dir)
        logger.info(self.hps)
        writer = SummaryWriter(log_dir=self.hps.data.exp_dir)
        writer_eval = SummaryWriter(
            log_dir=os.path.join(self.hps.data.exp_dir, 'eval'))

        dist.init_process_group(
            backend='gloo' if os.name == 'nt' or not self.device == 'cuda' else 'nccl',
            init_method='env://',
            world_size=1,
            rank=0,  # 1 gpu
        )
        torch.manual_seed(self.hps.train.seed)

        train_dataset = TextAudioSpeakerLoader(self.hps.data)
        train_sampler = DistributedBucketSampler(
            train_dataset,
            self.hps.train.batch_size,
            [
                32,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1100,
                1200,
                1300,
                1400,
                1500,
                1600,
                1700,
                1800,
                1900,
            ],
            num_replicas=1,  # 1 gpu
            rank=0,  # 1 gpu
            shuffle=True,
        )
        collate_fn = TextAudioSpeakerCollate()
        train_loader = DataLoader(
            train_dataset,
            num_workers=6,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=16,
        )

        net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(self.device)

        net_d = MultiPeriodDiscriminator(
            self.hps.model.use_spectral_norm).to(self.device)

        for name, param in net_g.named_parameters():
            if not param.requires_grad:
                print(f'{name} does not require grad')

        te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
        et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
        mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
        base_params = filter(
            lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
            net_g.parameters(),
        )

        optim_g = torch.optim.AdamW(
            [
                {'params': base_params, 'lr': self.hps.train.learning_rate},
                {
                    'params': net_g.enc_p.text_embedding.parameters(),
                    'lr': self.hps.train.learning_rate * self.hps.train.text_low_lr_rate,
                },
                {
                    'params': net_g.enc_p.encoder_text.parameters(),
                    'lr': self.hps.train.learning_rate * self.hps.train.text_low_lr_rate,
                },
                {
                    'params': net_g.enc_p.mrte.parameters(),
                    'lr': self.hps.train.learning_rate * self.hps.train.text_low_lr_rate,
                },
            ],
            self.hps.train.learning_rate,
            betas=self.hps.train.betas,
            eps=self.hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            self.hps.train.learning_rate,
            betas=self.hps.train.betas,
            eps=self.hps.train.eps,
        )

        if self.device == 'cuda':
            # rank = 0, 1 gpu
            net_g = DDP(net_g, device_ids=[0], find_unused_parameters=True)
            # rank = 0, 1 gpu
            net_d = DDP(net_d, device_ids=[0], find_unused_parameters=True)

        try:
            _, _, _, epoch_str = load_checkpoint(
                latest_checkpoint_path(
                    self.hps.s2_ckpt_dir, 'G_*.pth'),
                net_g,
                optim_g
            )
            logger.info('Loaded G')

            _, _, _, epoch_str = load_checkpoint(
                latest_checkpoint_path(
                    self.hps.s2_ckpt_dir, 'D_*.pth'),
                net_d,
                optim_d
            )
            logger.info('Loaded D')

            self.global_step = (epoch_str - 1) * len(train_loader)
        except:  # If checkpoints don't exist, load pretrained
            epoch_str = 1
            self.global_step = 0
            if self.hps.train.pretrained_s2G != '':
                logger.info(
                    f'Loaded pretrained {self.hps.train.pretrained_s2G}')
                print(
                    net_g.module.load_state_dict(
                        torch.load(self.hps.train.pretrained_s2G,
                                   map_location='cpu')['weight'],
                        strict=False,
                    ) if hasattr(net_g, 'module') else net_g.load_state_dict(
                        torch.load(self.hps.train.pretrained_s2G,
                                   map_location='cpu')['weight'],
                        strict=False,
                    )
                )
            if self.hps.train.pretrained_s2D != '':
                logger.info(
                    f'Loaded pretrained {self.hps.train.pretrained_s2D}')
                print(
                    net_d.module.load_state_dict(
                        torch.load(self.hps.train.pretrained_s2D,
                                   map_location='cpu')['weight']
                    ) if hasattr(net_g, 'module') else net_d.load_state_dict(
                        torch.load(self.hps.train.pretrained_s2D,
                                   map_location='cpu')['weight']
                    )
                )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.hps.train.lr_decay, last_epoch=-1
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=self.hps.train.lr_decay, last_epoch=-1
        )

        for _ in range(epoch_str):
            scheduler_g.step()
            scheduler_d.step()

        scaler = GradScaler(enabled=self.hps.train.fp16_run)  # enabled = False

        for epoch in range(epoch_str, self.hps.train.epochs + 1):
            # rank = 0, 1 gpu
            self.train_and_evaluate(
                epoch,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
            )
            scheduler_g.step()
            scheduler_d.step()

            clear_gpu_cache()

    def train_and_evaluate(self, epoch, nets, optims, schedulers, scaler, loaders, logger, writers):
        net_g, net_d = nets
        optim_g, optim_d = optims
        # scheduler_g, scheduler_d = schedulers
        train_loader, eval_loader = loaders
        writer, writer_eval = writers

        train_loader.batch_sampler.set_epoch(epoch)

        net_g.train()
        net_d.train()

        for batch_idx, (
            ssl,
            ssl_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            text,
            text_lengths,
        ) in tqdm(enumerate(train_loader)):
            spec, spec_lengths = spec.to(
                self.device), spec_lengths.to(self.device)
            y, y_lengths = y.to(self.device), y_lengths.to(self.device)
            ssl = ssl.to(self.device)
            ssl.requires_grad = False
            # ssl_lengths = ssl_lengths.to(self.device)
            text, text_lengths = text.to(
                self.device), text_lengths.to(self.device)

            with autocast(enabled=self.hps.train.fp16_run):  # enabled = False
                (
                    y_hat,
                    kl_ssl,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    stats_ssl,
                ) = net_g(ssl, spec, spec_lengths, text, text_lengths)

                mel = spec_to_mel_torch(
                    spec,
                    self.hps.data.filter_length,
                    self.hps.data.n_mel_channels,
                    self.hps.data.sampling_rate,
                    self.hps.data.mel_fmin,
                    self.hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    self.hps.data.filter_length,
                    self.hps.data.n_mel_channels,
                    self.hps.data.sampling_rate,
                    self.hps.data.hop_length,
                    self.hps.data.win_length,
                    self.hps.data.mel_fmin,
                    self.hps.data.mel_fmax,
                )

                y = commons.slice_segments(
                    y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size
                )  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=self.hps.train.fp16_run):  # enabled = False
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * \
                        self.hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                      z_mask) * self.hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            # rank = 0, 1 gpu
            if self.global_step % self.hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen,
                          loss_fm, loss_mel, kl_ssl, loss_kl]
                logger.info(
                    'Train Epoch: {} [{:.0f}%]'.format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item()
                            for x in losses] + [self.global_step, lr])

                scalar_dict = {
                    'loss/g/total': loss_gen_all,
                    'loss/d/total': loss_disc_all,
                    'learning_rate': lr,
                    'grad_norm_d': grad_norm_d,
                    'grad_norm_g': grad_norm_g,
                }
                scalar_dict.update(
                    {
                        'loss/g/fm': loss_fm,
                        'loss/g/mel': loss_mel,
                        'loss/g/kl_ssl': kl_ssl,
                        'loss/g/kl': loss_kl,
                    }
                )

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

                image_dict = {
                    'slice/mel_org': plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    'slice/mel_gen': plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    'all/mel': plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    'all/stats_ssl': plot_spectrogram_to_numpy(
                        stats_ssl[0].data.cpu().numpy()
                    ),
                }

                summarize(
                    writer=writer,
                    global_step=self.global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            self.global_step += 1

        if epoch % self.hps.train.save_every_epoch == 0:  # rank = 0, 1 gpu
            if self.hps.train.if_save_latest == 0:
                save_checkpoint(
                    net_g,
                    optim_g,
                    self.hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        self.hps.s2_ckpt_dir, f'G_{self.global_step}.pth'
                    ),
                )
                save_checkpoint(
                    net_d,
                    optim_d,
                    self.hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        self.hps.s2_ckpt_dir, f'D_{self.global_step}.pth'
                    ),
                )
            else:
                save_checkpoint(
                    net_g,
                    optim_g,
                    self.hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        self.hps.s2_ckpt_dir, f'G_latest.pth'
                    ),
                )
                save_checkpoint(
                    net_d,
                    optim_d,
                    self.hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        self.hps.s2_ckpt_dir, f'D_latest.pth'
                    ),
                )
            if self.hps.train.if_save_every_weights:  # rank = 0, 1 gpu
                if hasattr(net_g, 'module'):
                    ckpt = net_g.module.state_dict()
                else:
                    ckpt = net_g.state_dict()
                logger.info(
                    'Saving ckpt s2_e{}:{}'.format(
                        epoch,
                        savee(
                            ckpt,
                            f's2_e{epoch}_s{self.global_step}',
                            epoch,
                            self.global_step,
                            self.hps
                        ),
                    )
                )

        logger.info(f'====> Epoch: {epoch}')
