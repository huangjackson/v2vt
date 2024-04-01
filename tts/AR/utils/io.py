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

import sys

import torch
import yaml


def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


def save_config_to_yaml(config, path):
    assert path.endswith(".yaml")
    with open(path, "w") as f:
        f.write(yaml.dump(config))
        f.close()


def write_args(args, path):
    args_dict = dict(
        (name, getattr(args, name)) for name in dir(args) if not name.startswith("_")
    )
    with open(path, "a") as args_file:
        args_file.write("==> torch version: {}\n".format(torch.__version__))
        args_file.write(
            "==> cudnn version: {}\n".format(torch.backends.cudnn.version())
        )
        args_file.write("==> Cmd:\n")
        args_file.write(str(sys.argv))
        args_file.write("\n==> args:\n")
        for k, v in sorted(args_dict.items()):
            args_file.write("  %s: %s\n" % (str(k), str(v)))
        args_file.close()
