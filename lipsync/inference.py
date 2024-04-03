import os

import torch

from .config import LipSyncModel

from .third_part.face3d.util.preprocess import align_img
from .third_part.face3d.util.load_mats import load_lm3d
from .third_part.face3d.extract_kp_videos import KeypointExtractor

from .third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from .third_part.GFPGAN.gfpgan import GFPGANer
from .third_part.ganimation_replicate.model.ganimation import GANimationModel

from .utils import audio
from .utils.ffhq_preprocess import Croper
from .utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from .utils.inference_utils import (Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, split_coeff,
                                    trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict)


# def options():
#    parser = argparse.ArgumentParser(
#        description='Inference code to lip-sync videos in the wild using Wav2Lip models')
#
#    parser.add_argument('--DNet_path', type=str, default='checkpoints/DNet.pt')
#    parser.add_argument('--LNet_path', type=str,
#                        default='checkpoints/LNet.pth')
#    parser.add_argument('--ENet_path', type=str,
#                        default='checkpoints/ENet.pth')
#    parser.add_argument('--face3d_net_path', type=str,
#                        default='checkpoints/face3d_pretrain_epoch_20.pth')
#    parser.add_argument(
#        '--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
#    parser.add_argument(
#        '--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
#    parser.add_argument('--exp_img', type=str,
#                        help='Expression template. neutral, smile or image path', default='neutral')
#    parser.add_argument('--outfile', type=str,
#                        help='Video path to save result')
#
#    parser.add_argument(
#        '--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
#    parser.add_argument('--pads', nargs='+', type=int, default=[
#                        0, 20, 0, 0], help='Padding (top, bottom, left, right). Please adjust to include chin at least')
#    parser.add_argument('--face_det_batch_size', type=int,
#                        help='Batch size for face detection', default=4)
#    parser.add_argument('--LNet_batch_size', type=int,
#                        help='Batch size for LNet', default=16)
#    parser.add_argument('--img_size', type=int, default=384)
#    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
#                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
#                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
#    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
#                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
#                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
#    parser.add_argument('--nosmooth', default=False, action='store_true',
#                        help='Prevent smoothing face detections over a short temporal window')
#    parser.add_argument('--static', default=False, action='store_true')
#
#    parser.add_argument('--up_face', default='original')
#    parser.add_argument('--one_shot', action='store_true')
#    parser.add_argument('--without_rl1', default=False,
#                        action='store_true', help='Do not use the relative l1')
#    parser.add_argument('--tmp_dir', type=str, default='temp',
#                        help='Folder to save tmp results')
#    parser.add_argument('--re_preprocess', action='store_true')
#
#    args = parser.parse_args()
#    return args

class LipSyncInference:
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

        enhancer = FaceEnhancement(base_dir=self.model.models_dir, size=512, model='GPEN-BFR-512',
                                   use_sr=False, sr_model='rrdb_realesrnet_psnr', channel_multiplier=2,
                                   narrow=1, device=self.device)
        restorer = GFPGANer(model_path=os.path.join(self.model.models_dir, 'GFPGANv1.3.pth'),
                            upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
