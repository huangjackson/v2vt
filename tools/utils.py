import os
import sys
import subprocess
import warnings
import shutil

from huggingface_hub import snapshot_download


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


def download_model_from_hf(repo_id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_folder = snapshot_download(repo_id=repo_id)

    exclude_files = ['.gitattributes', 'README.md']

    for dirpath, dirnames, filenames in os.walk(model_folder):
        for filename in filenames:
            if filename in exclude_files:
                continue
            file_path = os.path.join(dirpath, filename)

            # Snapshots are symlinks, so we need to resolve the real path
            real_path = os.path.realpath(file_path)
            relative_dir = os.path.relpath(dirpath, model_folder)
            dest_dir = os.path.join(output_dir, relative_dir)

            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(real_path, os.path.join(dest_dir, filename))
            print(
                f'Copying {real_path} to {os.path.join(dest_dir, filename)}')

    print(f'Downloaded model to {output_dir}')


def check_models_and_install():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    nmt_models_path = os.path.join(script_dir, '../nmt/models')
    vr_models_path = os.path.join(script_dir, '../vr/models')
    tts_models_path = os.path.join(script_dir, '../tts/models/pretrained')

    nmt_required_files = ['model.bin', 'config.json',
                          'shared_vocabulary.json', 'source.spm', 'target.spm']
    vr_required_files = ['Kim_Vocal_2.onnx']
    tts_required_files = ['s2G488k.pth',
                          's1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt',
                          'chinese-hubert-base/config.json',
                          'chinese-hubert-base/preprocessor_config.json',
                          'chinese-hubert-base/pytorch_model.bin',
                          'chinese-roberta-wwm-ext-large/config.json',
                          'chinese-roberta-wwm-ext-large/tokenizer.json',
                          'chinese-roberta-wwm-ext-large/pytorch_model.bin']

    # Check for translation models
    for model_folder in os.listdir(nmt_models_path):
        model_path = os.path.join(nmt_models_path, model_folder)
        if os.path.isdir(model_path):
            if not all(os.path.exists(os.path.join(model_path, file)) for file in nmt_required_files):
                print('Translation models not found. Downloading from HuggingFace...')
                download_model_from_hf(
                    'huangjackson/ct2-opus-mt', nmt_models_path)

    # Check for vocal removal model
    if not all(os.path.exists(os.path.join(vr_models_path, file)) for file in vr_required_files):
        print('Vocal removal model not found. Downloading from HuggingFace...')
        download_model_from_hf('huangjackson/Kim_Vocal_2', vr_models_path)

    # Check for TTS models
    if not all(os.path.exists(os.path.join(tts_models_path, file)) for file in tts_required_files):
        print('TTS pretrained models not found. Downloading from HuggingFace...')
        download_model_from_hf('lj1995/GPT-SoVITS', tts_models_path)
