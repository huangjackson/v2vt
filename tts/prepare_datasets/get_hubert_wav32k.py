# Modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

import os

import numpy as np
import torch
import librosa
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor
)
from scipy.io import wavfile

from ..config import TTSModel

# TODO: Absolute import unlike others, make every import absolute?
from tools.ffmpeg import load_audio


class GetHubertWav32k:

    def __init__(self):
        self.model = TTSModel()

        self.hubert_dir = os.path.join(self.model.preproc_dir, 'hubert')
        self.wav32_dir = os.path.join(self.model.preproc_dir, 'wav32k')

        self.maxx = 0.95
        self.alpha = 0.5

        self.nan_fails = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.hubert_model = HubertModel.from_pretrained(
                self.model.hubert_path).to(self.device)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model.hubert_path)

            self.hubert_model.eval()
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

        ssl = self.hubert_model(tensor_wav16.unsqueeze(0))[
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
        os.makedirs(self.model.preproc_dir, exist_ok=True)
        os.makedirs(self.hubert_dir, exist_ok=True)
        os.makedirs(self.wav32_dir, exist_ok=True)

        with open(self.model.transcript_path, 'r', encoding='utf8') as f:
            lines = f.read().strip('\n').split('\n')

        for line in lines:
            try:
                wav_path, spk_name, language, text = line.split('|')
                self.name2go(wav_path)
            except Exception as e:
                return print(f'Error while processing {line}: {e}')
