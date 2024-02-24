# Includes modified code from https://github.com/Anjok07/ultimatevocalremovergui

import torch
import os
import traceback
import gc
from glob import glob

mdx_models_dir = "models"
mdx_cache_source_mapper = {}


class ModelData():
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = self.get_mdx_model_path()
        self.model_basename = os.path.splitext(
            os.path.basename(self.model_path))[0]

    def get_mdx_model_path(self):
        for file_name in glob(os.path.join(mdx_models_dir, '**/*.onnx'), recursive=True):
            if self.model_name in file_name:
                return file_name

        return ''


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()


def clear_cached_sources():
    mdx_cache_source_mapper = {}


def cached_source_callback(model_name=None):
    model, sources = None, None

    mapper = mdx_cache_source_mapper

    for key, value in mapper.items():
        if model_name in key:
            model = key
            sources = value

    return model, sources


def cached_model_source_holder(sources, model_name=None):
    mdx_cache_source_mapper = {
        **mdx_cache_source_mapper, **{model_name: sources}}


def execute_uvr(input_folder, output_folder, model_name="Kim_Vocal_2", agg=10):
    input_paths = [os.path.join(input_folder, name)
                   for name in os.listdir(input_folder)]

    try:
        model = ModelData("Kim_Vocal_2")

        for iteration, audio_file in enumerate(input_paths, start=1):
            clear_cached_sources()

            audio_file_base = f"{iteration}_{os.path.splitext(os.path.basename(audio_file))[0]}"

            process_data = {
                'model_data': model,
                'export_path': output_folder,
                'audio_file_base': audio_file_base,
                'audio_file': audio_file,
                'process_iteration': iteration,
                'cached_source_callback': cached_source_callback,
                'cached_model_source_holder': cached_model_source_holder,
            }

            separator = SeparateMDX(model, process_data)

            separator.separate()

            clear_gpu_cache()

        print("Process complete")

        clear_cached_sources()

    except Exception:
        print(traceback.format_exc())

        clear_cached_sources()
