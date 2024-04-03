# Copyright (c) 2023 OpenTalker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF

from tqdm import tqdm
from PIL import Image
from scipy.spatial import ConvexHull

from ..models import load_network, load_DNet
from ..third_part import face_detection
from ..third_part.face3d.models import networks

import warnings
warnings.filterwarnings("ignore")


exp_aus_dict = {        # AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r.
    'sad': torch.Tensor([[0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]]),
    'angry': torch.Tensor([[0,     0,      0.3,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]]),
    'surprise': torch.Tensor([[0, 0,      0,      0.2,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]])
}


def mask_postprocess(mask, thres=20):
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0
    mask = cv2.GaussianBlur(mask, (101, 101), 11)
    mask = cv2.GaussianBlur(mask, (101, 101), 11)
    return mask.astype(np.float32)


def trans_image(image):
    image = TF.resize(
        image, size=256, interpolation=Image.BICUBIC)
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return image


def obtain_seq_index(index, num_frames):
    seq = list(range(index-13, index+13))
    seq = [min(max(item, 0), num_frames-1) for item in seq]
    return seq


def transform_semantic(semantic, frame_index, crop_norm_ratio=None):
    index = obtain_seq_index(frame_index, semantic.shape[0])

    coeff_3dmm = semantic[index, ...]
    ex_coeff = coeff_3dmm[:, 80:144]  # expression # 64
    angles = coeff_3dmm[:, 224:227]  # euler angles for pose
    translation = coeff_3dmm[:, 254:257]  # translation
    crop = coeff_3dmm[:, 259:262]  # crop param

    if crop_norm_ratio:
        crop[:, -3] = crop[:, -3] * crop_norm_ratio

    coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
    return torch.Tensor(coeff_3dmm).permute(1, 0)


def find_crop_norm_ratio(source_coeff, target_coeffs):
    alpha = 0.3
    # mean different exp
    exp_diff = np.mean(
        np.abs(target_coeffs[:, 80:144] - source_coeff[:, 80:144]), 1)
    angle_diff = np.mean(np.abs(
        target_coeffs[:, 224:227] - source_coeff[:, 224:227]), 1)  # mean different angle
    # find the smallerest index
    index = np.argmin(alpha*exp_diff + (1-alpha)*angle_diff)
    crop_norm_ratio = source_coeff[:, -3] / target_coeffs[index:index+1, -3]
    return crop_norm_ratio


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, args, jaw_correction=False, detector=None):
    if detector == None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False, device=device)

    batch_size = args.face_det_batch_size
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), desc='FaceDet:'):
                predictions.extend(detector.get_detections_for_batch(
                    np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads if jaw_correction else (0, 20, 0, 0)
    for rect, image in zip(predictions, images):
        if rect is None:
            # check this frame where the face was not detected.
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError(
                'Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)]
               for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    torch.cuda.empty_cache()
    return results


def _load(checkpoint_path, device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpA = [gpA[num_levels-1]]
    lpB = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1, 0, -1):
        # Laplacian: subtract upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        gm = gm[:, :, np.newaxis]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


def load_model(args, device):
    D_Net = load_DNet(args).to(device)
    model = load_network(args).to(device)
    return D_Net, model


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(
            kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(
            kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}
    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(
                jacobian_diff, kp_source['jacobian'])
    return kp_new


def load_face3d_net(ckpt_path, device):
    net_recon = networks.define_net_recon(
        net_recon='resnet50', use_last_fc=False, init_path='').to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    net_recon.load_state_dict(checkpoint['net_recon'])
    net_recon.eval()
    return net_recon
