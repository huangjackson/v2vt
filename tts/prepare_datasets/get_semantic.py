# Modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/prepare_datasets/3-get-semantic.py

import os

import torch

from ..config import TTSModel
from ..utils import get_hparams_from_file
from ..module.models import SynthesizerTrn


class GetSemantic:

    def __init__(self):
        self.model = TTSModel()

        self.hubert_dir = os.path.join(self.model.preproc_dir, 'hubert')
        self.semantic_path = os.path.join(
            self.model.preproc_dir, 'semantic.tsv')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.hps = get_hparams_from_file(self.model.s2_config_path)
            self.vq_model = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            ).to(self.device)

            self.vq_model.eval()
            self.vq_model.load_state_dict(
                torch.load(self.model.pretrained_s2G_path,
                           map_location='cpu')['weight'],
                strict=False
            )
        except Exception as e:
            raise Exception(f'Error while loading model: {e}')

    def name2go(self, wav_name, lines):
        hubert_path = os.path.join(self.hubert_dir, f'{wav_name}.pt')

        if not os.path.exists(hubert_path):
            return print(f'HuBERT file not found: {hubert_path}')

        ssl_content = torch.load(hubert_path).to(self.device)

        codes = self.vq_model.extract_latent(ssl_content)
        semantic = ' '.join([str(c) for c in codes[0, 0, :].tolist()])
        lines.append(f'{wav_name}\t{semantic}')

    def execute(self):
        os.makedirs(self.model.preproc_dir, exist_ok=True)

        with open(self.model.transcript_path, 'r', encoding='utf8') as f:
            lines = f.read().strip('\n').split('\n')

        lines1 = []
        for line in lines:
            try:
                wav_path, spk_name, language, text = line.split('|')
                wav_name = os.path.basename(wav_path)

                self.name2go(wav_name, lines1)
            except Exception as e:
                return print(f'Error while processing {line}: {e}')

        with open(self.semantic_path, 'w', encoding='utf8') as f:
            f.write('\n'.join(lines1) + '\n')
