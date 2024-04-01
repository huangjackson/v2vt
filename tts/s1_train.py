# MIT License
#
# Copyright (c) 2024 RVC-Boss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import logging
from collections import OrderedDict
from pathlib import Path
from random import randint

import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from .config import ModelData
from .AR.utils.io import load_yaml_config
from .AR.utils import get_newest_ckpt
from .AR.models.t2s_lightning_module import Text2SemanticLightningModule
from .AR.data.data_module import Text2SemanticDataModule

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

torch.set_float32_matmul_precision('high')


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        # if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if self.if_save_latest:  # if if_save_latest, clear all previous ckpts after saving next
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest:
                    for name in to_clean:
                        try:
                            os.remove(os.path.join(self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od['weight'] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od['weight'][key] = dictt[key].half()
                    to_save_od['config'] = self.config
                    to_save_od['info'] = f'GPT-e{trainer.current_epoch + 1}'
                    torch.save(
                        to_save_od,
                        os.path.join(self.half_weights_save_dir,
                                     f's1-e{trainer.current_epoch + 1}.ckpt')
                    )
            self._save_last_checkpoint(trainer, monitor_candidates)


class S1Train:

    def __init__(self, batch_size, total_epoch, if_dpo, if_save_latest,
                 if_save_every_weights, save_every_epoch):
        self.model = ModelData()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            os.makedirs(self.model.tmp_dir, exist_ok=True)
            os.makedirs(self.model.s1_dir, exist_ok=True)
            os.makedirs(self.model.s1_ckpt_dir, exist_ok=True)
            os.makedirs(self.model.gpt_weights_path, exist_ok=True)

            with open(self.model.s1_config_path, 'r') as f:
                data = f.read()
                data = yaml.load(data, Loader=yaml.FullLoader)

            data['train']['precision'] = '32'  # 1 gpu
            data['train']['batch_size'] = max(1, batch_size // 2)  # 1 gpu
            data['train']['epochs'] = total_epoch
            data['pretrained_s1'] = self.model.pretrained_s1_path
            data['train']['save_every_n_epoch'] = save_every_epoch
            data['train']['if_dpo'] = if_dpo
            data['train']['if_save_latest'] = if_save_latest
            data['train']['if_save_every_weights'] = if_save_every_weights
            data['train']['half_weights_save_dir'] = self.model.gpt_weights_path
            data['train_semantic_path'] = os.path.join(
                self.model.preproc_dir, 'semantic.tsv')
            data['train_phoneme_path'] = os.path.join(
                self.model.preproc_dir, 'phoneme.txt')
            data['output_dir'] = self.model.s1_dir

            self.tmp_config_path = os.path.join(
                self.model.tmp_dir, 'tmp_s1.yaml')
            with open(self.tmp_config_path, 'w') as f:
                f.write(yaml.dump(data, default_flow_style=False))
        except Exception as e:
            raise Exception(f'Error creating temporary config file: {e}')

        try:
            self.config = load_yaml_config(self.tmp_config_path)
        except Exception as e:
            raise Exception(
                f'Error loading config from temporary config file: {e}')

    def run(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(randint(10000, 20000))

        output_dir = Path(self.config['output_dir'])
        ckpt_dir = output_dir / 'ckpt'

        seed_everything(self.config['train']['seed'], workers=True)
        ckpt_callback: ModelCheckpoint = my_model_ckpt(
            config=self.config,
            if_save_latest=self.config['train']['if_save_latest'],
            if_save_every_weights=self.config['train']['if_save_every_weights'],
            half_weights_save_dir=self.config['train']['half_weights_save_dir'],
            save_top_k=-1,
            monitor='top_3_acc',
            mode='max',
            save_on_train_epoch_end=True,
            every_n_epochs=self.config['train']['save_every_n_epoch'],
            dirpath=ckpt_dir,
        )

        logger = TensorBoardLogger(name='logs', save_dir=output_dir)

        trainer: Trainer = Trainer(
            max_epochs=self.config['train']['epochs'],
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            limit_val_batches=0,
            devices=-1 if self.device == 'cuda' else 1,
            benchmark=False,
            fast_dev_run=False,
            strategy=DDPStrategy(
                process_group_backend='gloo' if os.name == 'nt' else 'nccl',
            ) if self.device == 'cuda' else 'auto',
            precision=self.config['train']['precision'],
            logger=logger,
            num_sanity_val_steps=0,
            callbacks=[ckpt_callback],
        )

        model: Text2SemanticLightningModule = Text2SemanticLightningModule(
            self.config, output_dir
        )

        data_module: Text2SemanticDataModule = Text2SemanticDataModule(
            self.config,
            train_semantic_path=self.config['train_semantic_path'],
            train_phoneme_path=self.config['train_phoneme_path'],
        )

        try:
            # Use regex to match numbers in filenames and sort
            newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
            ckpt_path = ckpt_dir / newest_ckpt_name
        except Exception as e:
            ckpt_path = None
            print(f'Error while getting latest checkpoint: {e}')

        print('ckpt_path: ', ckpt_path)
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
