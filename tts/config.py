import os

script_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(script_dir, 'models')
pretrained_models_dir = os.path.join(models_dir, 'pretrained')


class ModelData:

    def __init__(self):
        self.sovits_path = ''
        self.gpt_path = ''

        try:
            self.hubert_path = self.get_hubert_model_path()
            self.roberta_path = self.get_roberta_model_path()
            self.pretrained_sovits_path = self.get_sovits_model_path()
            self.pretrained_gpt_path = self.get_gpt_model_path()
        except Exception as e:
            raise Exception(f'Error while getting pretrained models: {e}')

    def get_hubert_model_path(self):
        model_path = os.path.join(pretrained_models_dir, 'chinese-hubert-base')
        required_files = ['config.json',
                          'preprocessor_config.json', 'pytorch_model.bin']

        if not all(os.path.exists(os.path.join(model_path, file)) for file in required_files):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path

    def get_roberta_model_path(self):
        model_path = os.path.join(
            pretrained_models_dir, 'chinese-roberta-wwm-ext-large')
        required_files = ['config.json',
                          'tokenizer.json', 'pytorch_model.bin']

        if not all(os.path.exists(os.path.join(model_path, file)) for file in required_files):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path

    def get_sovits_model_path(self):
        model_path = os.path.join(pretrained_models_dir, 's2G488k.pth')

        if not os.path.exists(model_path):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path

    def get_gpt_model_path(self):
        model_path = os.path.join(
            pretrained_models_dir, 's1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt')

        if not os.path.exists(model_path):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path
