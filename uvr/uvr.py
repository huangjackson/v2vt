# Includes modified code from https://github.com/Anjok07/ultimatevocalremovergui

import os
import argparse
from glob import glob

from separate import (
    SeparateMDX, verify_audio, clear_gpu_cache
)

script_dir = os.path.dirname(os.path.realpath(__file__))
mdx_models_dir = os.path.join(script_dir, 'models')


class ModelData():

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = self.get_mdx_model_path()
        self.model_basename = os.path.splitext(
            os.path.basename(self.model_path))[0]

        # Settings specific to Kim_Vocal_2.onnx model
        self.compensate = 1.009
        self.mdx_dim_f_set = 3072
        self.mdx_dim_t_set = 8
        self.mdx_n_fft_scale_set = 6144
        self.primary_stem = 'Vocals'

        self.mdx_segment_size = 256
        self.mdx_batch_size = 1

        if not self.model_path:
            raise ValueError(
                "Model not found. Please check the model name and path.")

    def get_mdx_model_path(self):
        for file_name in glob(os.path.join(mdx_models_dir, '**/*.onnx'), recursive=True):
            if self.model_name in file_name:
                return file_name

        return ''


def execute_uvr(input_file, output_folder, model_name='Kim_Vocal_2'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        model = ModelData(model_name)

        if not verify_audio(input_file):
            return print(
                f'{os.path.basename(input_file)} is not a valid .wav file')

        audio_file_base = os.path.splitext(os.path.basename(input_file))[0]

        process_data = {
            'export_path': output_folder,
            'audio_file_base': audio_file_base,
            'audio_file': input_file,
        }

        separator = SeparateMDX(model, process_data)

        separator.separate()

        clear_gpu_cache()

    except Exception as e:
        return print(f'An error occured during vocal removal: {e}')

    return output_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to the source WAV file')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Output folder to store isolated vocals')
    # parser.add_argument('-m', '--model_name', type=str, default='Kim_Vocal_2',
    #                     help='Name of model to use for vocal separation')

    args = parser.parse_args()

    output_file_path = execute_uvr(
        args.input_file, args.output_folder)

    print(f'UVR complete - files written to {output_file_path}')
