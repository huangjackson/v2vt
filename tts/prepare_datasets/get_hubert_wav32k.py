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

import numpy as np
import torch
import librosa
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor
)
from scipy.io import wavfile

# TODO: Absolute import unlike others, make every import absolute?
from tools.ffmpeg import load_audio


class GetHubertWav32k:

    def __init__(self, transcribed_file, sliced_audio_folder, output_folder, hubert_path):
        self.transcribed_file = transcribed_file
        self.sliced_audio_folder = sliced_audio_folder
        self.output_folder = output_folder
        self.hubert_path = hubert_path

        self.hubert_dir = os.path.join(self.output_folder, 'hubert')
        self.wav32_dir = os.path.join(self.output_folder, 'wav32k')

        self.maxx = 0.95
        self.alpha = 0.5

        self.nan_fails = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.model = HubertModel.from_pretrained(
                self.hubert_path).to(self.device)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.hubert_path)

            self.model.eval()
        except Exception as e:
            raise Exception(f'Error while loading hubert model: {e}')

    def name2go(self, wav_path):
        wav_name = os.path.basename(wav_path)

        tmp_audio = load_audio(wav_path, 32000)
        tmp_max = np.abs(tmp_audio).max()

        if tmp_max > 2.2:
            return print(f'{wav_name} has max {tmp_max}, filtered')

        tmp_audio32 = (tmp_audio / tmp_max * (self.maxx *
                       self.alpha*32768)) + ((1 - self.alpha)*32768) * tmp_audio
        tmp_audio32b = (tmp_audio / tmp_max * (self.maxx *
                        self.alpha*32768)) + ((1 - self.alpha)*1145.14) * tmp_audio
        tmp_audio = librosa.resample(
            tmp_audio32b, orig_sr=32000, target_sr=16000
        )

        tensor_wav16 = torch.from_numpy(tmp_audio).to(self.device)

        ssl = self.model(tensor_wav16.unsqueeze(0))[
            'last_hidden_state'].transpose(1, 2).cpu()
        if np.isnan(ssl.detach().numpy()).any():
            self.nan_fails.append(wav_name)
            return print(f'{wav_name} has nan, filtered')
        wavfile.write(
            f'{self.wav32_dir}/{wav_name}',
            32000,
            tmp_audio32.astype('int16'),
        )
        torch.save(ssl, os.path.join(self.hubert_dir, f'{wav_name}.pt'))

    def execute(self):
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.hubert_dir, exist_ok=True)
        os.makedirs(self.wav32_dir, exist_ok=True)

        with open(self.transcribed_file, 'r', encoding='utf8') as f:
            lines = f.read().strip('\n').split('\n')

        for line in lines:
            try:
                wav_path, spk_name, language, text = line.split('|')
                self.name2go(wav_path)
            except Exception as e:
                return print(f'Error while processing {line}: {e}')
