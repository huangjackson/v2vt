import os


class TTSModel:

    def __init__(self):
        # Constants
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = os.path.join(self.script_dir, '..')
        self.configs_dir = os.path.join(self.script_dir, 'configs')
        self.models_dir = os.path.join(self.script_dir, 'models')

        # Default output directories
        self.tmp_dir = os.path.join(self.project_dir, 'TEMP')
        self.out_dir = os.path.join(self.project_dir, 'logs')
        self.preproc_dir = os.path.join(self.out_dir, '1-preproc')
        self.s2_dir = os.path.join(self.out_dir, '2-train-s2')
        self.s2_ckpt_dir = os.path.join(self.s2_dir, 'ckpt')
        self.s1_dir = os.path.join(self.out_dir, '3-train-s1')
        self.s1_ckpt_dir = os.path.join(self.s1_dir, 'ckpt')
        self.sovits_weights_path = os.path.join(
            self.out_dir, 'SoVITS_weights')
        self.gpt_weights_path = os.path.join(self.out_dir, 'GPT_weights')

        # Preprocessed data paths
        self.transcript_path = os.path.join(self.out_dir, 'transcript.list')
        self.s2_config_path = os.path.join(self.configs_dir, 's2.json')
        self.s1_config_path = os.path.join(self.configs_dir, 's1longer.yaml')

        try:
            # Get pretrained models paths
            self.hubert_path = self.get_model_folder(
                model_name='chinese-hubert-base',
                required_files=['config.json',
                                'preprocessor_config.json',
                                'pytorch_model.bin']
            )
            self.roberta_path = self.get_model_folder(
                model_name='chinese-roberta-wwm-ext-large',
                required_files=['config.json',
                                'tokenizer.json',
                                'pytorch_model.bin']
            )

            # Get pretrained SoVITS paths
            self.pretrained_s2G_path = self.get_model_path('s2G488k.pth')
            self.pretrained_s2D_path = self.get_model_path('s2D488k.pth')

            # Get pretrained GPT path
            self.pretrained_s1_path = self.get_model_path(
                's1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt')
        except Exception as e:
            raise Exception(f'Error while getting pretrained models: {e}')

    def get_model_path(self, model_name):
        model_path = os.path.join(self.models_dir, model_name)

        if not os.path.exists(model_path):
            raise Exception(
                f'{model_name} model file is missing. Please check the models directory.')

        return model_path

    def get_model_folder(self, model_name, required_files):
        model_path = os.path.join(self.models_dir, model_name)

        if not all(os.path.exists(os.path.join(model_path, file)) for file in required_files):
            raise Exception(
                f'One or more {model_name} model files are missing. Please check the models directory.')

        return model_path
