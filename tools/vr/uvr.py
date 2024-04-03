import os

from .separate import (
    SeparateMDX, verify_audio, clear_gpu_cache
)


class ModelData:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = self.get_model_path()
        self.model_basename = os.path.splitext(
            os.path.basename(self.model_path))[0]

        # Settings specific to Kim_Vocal_2.onnx model
        # TODO: replace with data from model config.json
        self.compensate = 1.009
        self.mdx_dim_f_set = 3072
        self.mdx_dim_t_set = 8
        self.mdx_n_fft_scale_set = 6144
        self.primary_stem = 'Vocals'

        self.mdx_segment_size = 256
        self.mdx_batch_size = 1

    def get_model_path(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, 'models', self.model_name)
        if not os.path.exists(model_path):
            raise Exception(
                'Model not found. Please check the models directory.')
        return model_path


class UltimateVocalRemover:

    def __init__(self, input_file, output_folder, model_name='Kim_Vocal_2.onnx'):
        self.input_file = input_file
        self.output_folder = output_folder
        self.model_name = model_name

    def execute(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        try:
            model = ModelData(self.model_name)

            if not verify_audio(self.input_file):
                return print(
                    f'{os.path.basename(self.input_file)} is not a valid .wav file')

            audio_file_base = os.path.splitext(
                os.path.basename(self.input_file))[0]

            process_data = {
                'export_path': self.output_folder,
                'audio_file_base': audio_file_base,
                'audio_file': self.input_file,
            }

            separator = SeparateMDX(model, process_data)
            separator.separate()
            clear_gpu_cache()

        except Exception as e:
            return print(f'An error occurred during vocal removal: {e}')

        return self.output_folder
