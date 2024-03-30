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

import torch

from ..utils import get_hparams_from_file
from ..module.models import SynthesizerTrn


class GetSemantic:

    def __init__(self, transcribed_file, output_folder, pretrained_s2G_path, s2_config_path):
        self.transcribed_file = transcribed_file
        self.output_folder = output_folder
        self.pretrained_s2G_path = pretrained_s2G_path
        self.s2_config_path = s2_config_path

        self.hubert_dir = os.path.join(self.output_folder, 'hubert')
        self.semantic_path = os.path.join(self.output_folder, 'semantic.tsv')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.hps = get_hparams_from_file(self.s2_config_path)
            self.vq_model = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            ).to(self.device)

            self.vq_model.eval()
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
        os.makedirs(self.output_folder, exist_ok=True)

        with open(self.transcribed_file, 'r', encoding='utf8') as f:
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
