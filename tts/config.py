import os


class ModelData:

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = os.path.join(self.script_dir, '..')
        self.configs_dir = os.path.join(self.script_dir, 'configs')
        self.models_dir = os.path.join(self.script_dir, 'models')

        self.tmp_dir = os.path.join(self.project_dir, 'TEMP')
        self.out_dir = os.path.join(self.project_dir, 'logs')

        # TODO: Assign path to models (SoVITS: v2vt/logs/2-train-s2)
        self.sovits_path = ''
        self.gpt_path = ''

        self.sovits_weights_path = os.path.join(
            self.out_dir, 'SoVITS_weights')
        self.gpt_weights_path = os.path.join(self.out_dir, 'GPT_weights')

        self.s2_config_path = os.path.join(self.configs_dir, 's2.json')

        try:
            self.hubert_path = self.get_hubert_model_path()
            self.roberta_path = self.get_roberta_model_path()

            # Get SoVITS paths
            self.pretrained_s2G_path = self.get_s2G_model_path()
            self.pretrained_s2D_path = self.get_s2D_model_path()

            # Get GPT path
            self.pretrained_s1_path = self.get_s1_model_path()
        except Exception as e:
            raise Exception(f'Error while getting pretrained models: {e}')

    def get_hubert_model_path(self):
        model_path = os.path.join(
            self.models_dir, 'chinese-hubert-base')
        required_files = ['config.json',
                          'preprocessor_config.json', 'pytorch_model.bin']

        if not all(os.path.exists(os.path.join(model_path, file)) for file in required_files):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path

    def get_roberta_model_path(self):
        model_path = os.path.join(
            self.models_dir, 'chinese-roberta-wwm-ext-large')
        required_files = ['config.json',
                          'tokenizer.json', 'pytorch_model.bin']

        if not all(os.path.exists(os.path.join(model_path, file)) for file in required_files):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path

    def get_s2G_model_path(self):
        model_path = os.path.join(self.models_dir, 's2G488k.pth')

        if not os.path.exists(model_path):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path

    def get_s2D_model_path(self):
        model_path = os.path.join(self.models_dir, 's2D488k.pth')

        if not os.path.exists(model_path):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path

    def get_s1_model_path(self):
        model_path = os.path.join(
            self.models_dir, 's1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt')

        if not os.path.exists(model_path):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path
