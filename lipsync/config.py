import os


class LipSyncModel:

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = os.path.join(self.script_dir, '..')
        self.models_dir = os.path.join(self.script_dir, 'checkpoints')
        self.bfm_dir = os.path.join(self.models_dir, 'BFM')

        self.tmp_dir = os.path.join(self.project_dir, 'TEMP')
        self.out_dir = os.path.join(self.project_dir, 'logs')

        # TODO: let user set config
        self.fps = 25
        self.pads = [0, 20, 0, 0]
        self.face_det_batch_size = 4
        self.LNet_batch_size = 16
        self.img_size = 384
        self.crop = [0, -1, 0, -1]
        self.box = [-1, -1, -1, -1]
        self.nosmooth = False
        self.static = False
        self.exp_img = 'neutral'
        self.up_face = 'original'
        self.one_shot = False
        self.without_rl1 = False
        self.re_preprocess = False

        try:
            self.DNet_path = self.get_model_path('DNet.pt')
            self.LNet_path = self.get_model_path('LNet.pth')
            self.ENet_path = self.get_model_path('ENet.pth')
            self.face3d_net_path = self.get_model_path(
                'face3d_pretrain_epoch_20.pth')
            self.shape_predictor_path = self.get_model_path(
                'shape_predictor_68_face_landmarks.dat')
            self.expression_path = self.get_model_path('expression.mat')
        except Exception as e:
            raise Exception(f'Error while getting pretrained models: {e}')

    def get_model_path(self, model_name):
        model_path = os.path.join(self.models_dir, model_name)

        if not os.path.exists(model_path):
            raise Exception(
                f'{model_name} model file is missing. Please check the checkpoints directory.')

        return model_path
