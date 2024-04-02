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

import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

import torch
import logging

logging.getLogger("numba").setLevel(logging.WARNING)


cnhubert_base_path = None


class CNHubert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = HubertModel.from_pretrained(cnhubert_base_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            cnhubert_base_path
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats


# class CNHubertLarge(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = HubertModel.from_pretrained("/data/docker/liujing04/gpt-vits/chinese-hubert-large")
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/data/docker/liujing04/gpt-vits/chinese-hubert-large")
#     def forward(self, x):
#         input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
#         feats = self.model(input_values)["last_hidden_state"]
#         return feats
#
# class CVec(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = HubertModel.from_pretrained("/data/docker/liujing04/vc-webui-big/hubert_base")
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/data/docker/liujing04/vc-webui-big/hubert_base")
#     def forward(self, x):
#         input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
#         feats = self.model(input_values)["last_hidden_state"]
#         return feats
#
# class cnw2v2base(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = Wav2Vec2Model.from_pretrained("/data/docker/liujing04/gpt-vits/chinese-wav2vec2-base")
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/data/docker/liujing04/gpt-vits/chinese-wav2vec2-base")
#     def forward(self, x):
#         input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
#         feats = self.model(input_values)["last_hidden_state"]
#         return feats


def get_model():
    model = CNHubert()
    model.eval()
    return model


# def get_large_model():
#     model = CNHubertLarge()
#     model.eval()
#     return model
#
# def get_model_cvec():
#     model = CVec()
#     model.eval()
#     return model
#
# def get_model_cnw2v2base():
#     model = cnw2v2base()
#     model.eval()
#     return model


def get_content(hmodel, wav_16k_tensor):
    with torch.no_grad():
        feats = hmodel(wav_16k_tensor)
    return feats.transpose(1, 2)
