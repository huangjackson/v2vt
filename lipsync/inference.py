import os

import torch

from .config import LipSyncModel

from .third_part.face3d.util.preprocess import align_img
from .third_part.face3d.util.load_mats import load_lm3d
from .third_part.face3d.extract_kp_videos import KeypointExtractor

from .third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from .third_part.GFPGAN.gfpgan import GFPGANer
from .third_part.ganimation_replicate.model.ganimation import GANimationModel

from .utils.audio import load_wav
from .utils.ffhq_preprocess import Croper
from .utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from .utils.inference_utils import (Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, split_coeff,
                                    trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict)


class LipSyncInference:
    def __init__(self, input_video, input_audio):
        self.input_video = input_video
        self.input_audio = input_audio

        self.model = LipSyncModel()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self):
        os.makedirs(self.model.tmp_dir, exist_ok=True)
        os.makedirs(self.model.out_dir, exist_ok=True)

        enhancer = FaceEnhancement(base_dir=self.model.models_dir, size=512, model='GPEN-BFR-512',
                                   use_sr=False, sr_model='rrdb_realesrnet_psnr', channel_multiplier=2,
                                   narrow=1, device=self.device)
        restorer = GFPGANer(model_path=os.path.join(self.model.models_dir, 'GFPGANv1.3.pth'),
                            upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
