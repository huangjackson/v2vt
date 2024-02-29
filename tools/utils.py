import sys
import subprocess
import warnings


def is_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except FileNotFoundError:
        return False


def check_torch_and_cuda():
    try:
        import torch

        is_torch_installed = True

        is_cuda_available = torch.cuda.is_available()
    except ImportError:
        is_torch_installed = False
        is_cuda_available = False

    return is_torch_installed, is_cuda_available


def check_dependencies():
    ffmpeg = is_ffmpeg_installed()
    torch, cuda = check_torch_and_cuda()

    if not ffmpeg:
        print('FFmpeg is not installed, please check the Prerequisites section in README.md for installation')
    if not torch:
        print('PyTorch is not installed, please check the Prerequisites section in README.md for installation')

    if not ffmpeg or not torch:
        sys.exit(1)

    if not cuda:
        warnings.warn(
            'Warning: No CUDA-compatible GPU detected. An NVIDIA GPU is highly recommended.')
