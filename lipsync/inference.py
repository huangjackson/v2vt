import os

import torch

from .config import LipSyncModel
# from .third_part.GPEN.gpen_face_enhancer import FaceEnhancement
# from .third_part.GFPGAN.gfpgan import GFPGANer


class Inference:
    def __init__(self, input_video, input_audio, output_folder,
                 exp_img='neutral', up_face='original'):
        self.input_video = input_video
        self.input_audio = input_audio
        self.output_folder = output_folder
        self.exp_img = exp_img
        self.up_face = up_face

        self.model = LipSyncModel()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self):
        os.makedirs(self.model.tmp_dir, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        # enhancer = FaceEnhancement(base_dir=self.model.models_dir, size=512, model='GPEN-BFR-512',
        #                            use_sr=False, sr_model='rrdb_realesrnet_psnr', channel_multiplier=2,
        #                            narrow=1, device=self.device)
        # restorer = GFPGANer(model_path=os.path.join(self.model.models_dir, 'GFPGANv1.3.pth'),
        #                     upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
