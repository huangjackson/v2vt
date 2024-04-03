# Copyright (c) 2023 OpenTalker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from models.DNet import DNet
from models.LNet import LNet
from models.ENet import ENet


def _load(checkpoint_path):
    map_location = None if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"] if 'arcface' not in path else checkpoint
    new_s = {}
    for k, v in s.items():
        if 'low_res' in k:
            continue
        else:
            new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=False)
    return model


def load_network(args):
    L_net = LNet()
    L_net = load_checkpoint(args.LNet_path, L_net)
    E_net = ENet(lnet=L_net)
    model = load_checkpoint(args.ENet_path, E_net)
    return model.eval()


def load_DNet(args):
    D_Net = DNet()
    print("Load checkpoint from: {}".format(args.DNet_path))
    checkpoint = torch.load(
        args.DNet_path, map_location=lambda storage, loc: storage)
    D_Net.load_state_dict(checkpoint['net_G_ema'], strict=False)
    return D_Net.eval()
