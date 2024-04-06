# Modified from https://github.com/OpenTalker/video-retalking/blob/main/inference.py

import os
from PIL import Image

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from .config import LipSyncModel

from .third_part.face3d.util.preprocess import align_img
from .third_part.face3d.util.load_mats import load_lm3d
from .third_part.face3d.extract_kp_videos import KeypointExtractor

from .third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from .third_part.GFPGAN.gfpgan import GFPGANer
from .third_part.ganimation_replicate.model.ganimation import GANimationModel

from .utils.audio import load_wav, melspectrogram
from .utils.ffhq_preprocess import Croper
from .utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from .utils.inference_utils import (Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, split_coeff,
                                    trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict)

# TODO: Absolute import unlike others, make every import absolute?
from tools.ffmpeg import add_audio_to_video


class LipSyncInference:
    def __init__(self, input_video, input_audio, output_path):
        self.input_video = input_video
        self.input_audio = input_audio
        self.output_path = output_path

        self.model = LipSyncModel()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def datagen(self, frames, mels, full_frames, frames_pil, cox):
        img_batch = []
        ref_batch = []
        mel_batch = []
        frame_batch = []
        coords_batch = []
        full_frame_batch = []

        refs = []
        image_size = 256

        kp_extractor = KeypointExtractor()
        fr_pil = [Image.fromarray(frame) for frame in frames]

        x12_landmarks_path = os.path.join(
            self.model.tmp_dir, 'x12_landmarks.txt')
        lms = kp_extractor.extract_keypoint(fr_pil, x12_landmarks_path)
        frames_pil = [(lm, frame) for frame, lm in zip(fr_pil, lms)]
        crops, orig_images, quads = crop_faces(
            image_size, frames_pil, scale=1.0, use_fa=True)
        inverse_transforms = [calc_alignment_coefficients(
            quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
        del kp_extractor.detector

        oy1, oy2, ox1, ox2 = cox
        face_det_results = face_detect(
            full_frames, self.model, jaw_correction=True)

        for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames, face_det_results):
            imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
                cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

            ff = full_frame.copy()
            ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(
                np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))

            oface, coords = face_det
            y1, y2, x1, x2 = coords

            refs.append(ff[y1:y2, x1:x2])

        for i, m in enumerate(mels):
            idx = 0 if self.model.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face = refs[idx]
            oface, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.model.img_size, self.model.img_size))
            oface = cv2.resize(
                oface, (self.model.img_size, self.model.img_size))

            img_batch.append(oface)
            ref_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)
            full_frame_batch.append(full_frames[idx].copy())

            if len(img_batch) >= self.model.LNet_batch_size:
                img_batch = np.asarray(img_batch)
                mel_batch = np.asarray(mel_batch)
                ref_batch = np.asarray(ref_batch)

                img_masked = img_batch.copy()
                img_original = img_batch.copy()
                img_masked[:, self.model.img_size//2:] = 0
                img_batch = np.concatenate(
                    (img_masked, ref_batch), axis=3) / 255.
                mel_batch = np.reshape(
                    mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch

                img_batch = []
                mel_batch = []
                frame_batch = []
                coords_batch = []
                img_original = []
                full_frame_batch = []
                ref_batch = []

        if len(img_batch) > 0:
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)
            ref_batch = np.asarray(ref_batch)

            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, self.model.img_size//2:] = 0
            img_batch = np.concatenate(
                (img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch

    def run(self):
        if not os.path.isfile(self.input_video):
            raise ValueError('Input video file not found.')

        os.makedirs(self.model.tmp_dir, exist_ok=True)
        os.makedirs(self.model.out_dir, exist_ok=True)

        print(f'[Info] Using {self.device} for inference.')

        enhancer = FaceEnhancement(base_dir=self.model.models_dir, size=512, model='GPEN-BFR-512',
                                   use_sr=False, sr_model='rrdb_realesrnet_psnr', channel_multiplier=2,
                                   narrow=1, device=self.device)
        restorer = GFPGANer(model_path=os.path.join(self.model.models_dir, 'GFPGANv1.3.pth'),
                            upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

        if self.input_video.endswith(('.jpg', '.png', '.jpeg')):
            self.model.static = True
            full_frames = [cv2.imread(self.input_video)]
            fps = self.model.fps
        else:
            video_stream = cv2.VideoCapture(self.input_video)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                y1, y2, x1, x2 = self.model.crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)

        print(
            f'[Step 0] Number of frames available for inference: {len(full_frames)}')

        croper = Croper(self.model.shape_predictor_path)
        full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                           for frame in full_frames]
        full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)

        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly + \
            ly, min(cly + ry, full_frames[0].shape[0]), clx + \
            lx, min(clx + rx, full_frames_RGB[0].shape[1])

        frames_pil = [Image.fromarray(cv2.resize(frame, (256, 256)))
                      for frame in full_frames_RGB]

        landmarks_path = os.path.join(self.model.tmp_dir, 'landmarks.txt')
        if not os.path.isfile(landmarks_path) or self.model.re_preprocess:
            print('[Step 1] Extracting landmarks...')
            kp_extractor = KeypointExtractor()
            lm = kp_extractor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print('[Step 1] Using saved landmarks...')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(full_frames), -1, 2])

        coeffs_path = os.path.join(self.model.tmp_dir, 'coeffs.npy')
        if not os.path.isfile(coeffs_path) or self.model.exp_img is not None or self.model.re_preprocess:
            net_recon = load_face3d_net(
                self.model.face3d_net_path, self.device)
            lm3d_std = load_lm3d(self.model.bfm_dir)

            video_coeffs = []
            for idx in tqdm(range(len(frames_pil)), desc='[Step 2] Running 3DMM extraction'):
                frame = frames_pil[idx]
                W, H = frame.size
                lm_idx = lm[idx].reshape([-1, 2])
                if np.mean(lm_idx) == -1:
                    lm_idx = (lm3d_std[:, :2] + 1) / 2
                    lm_idx = np.concatenate(
                        [lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
                else:
                    lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

                trans_params, im_idx, lm_idx, _ = align_img(
                    frame, lm_idx, lm3d_std)
                trans_params = np.array(
                    [float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)

                im_idx_tensor = torch.tensor(np.array(
                    im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    coeffs = split_coeff(net_recon(im_idx_tensor))

                pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
                pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'],
                                            pred_coeff['angle'], pred_coeff['gamma'], pred_coeff['trans'],
                                            trans_params[None]], 1)

                video_coeffs.append(pred_coeff)

            semantic_npy = np.array(video_coeffs)[:, 0]
            np.save(coeffs_path, semantic_npy)
        else:
            print('[Step 2] Using saved coefficients...')
            semantic_npy = np.load(coeffs_path).astype(np.float32)

        temp_path = os.path.join(self.model.tmp_dir, 'temp.txt')
        if self.model.exp_img is not None and ('.png' in self.model.exp_img or '.jpg' in self.model.exp_img):
            print(f'Extracting exp from {self.model.exp_img}')
            exp_pil = Image.open(self.model.exp_img).convert('RGB')
            # TODO: Add models for expression extraction from image
            lm3d_std = load_lm3d('third_part/face3d/BFM')

            W, H = exp_pil.size
            kp_extractor = KeypointExtractor()
            lm_exp = kp_extractor.extract_keypoint([exp_pil], temp_path)[0]

            if np.mean(lm_exp) == -1:
                lm_exp = (lm3d_std[:, :2] + 1) / 2
                lm_exp = np.concatenate(
                    [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
            else:
                lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

            trans_params, im_exp, lm_exp, _ = align_img(
                exp_pil, lm_exp, lm3d_std)
            trans_params = np.array(
                [float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)

            im_exp_tensor = torch.tensor(np.array(
                im_exp)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
            with torch.no_grad():
                expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
            del net_recon
        elif self.model.exp_img == 'smile':
            expression = torch.tensor(loadmat(self.model.expression_path)[
                                      'expression_mouth'])[0]
        else:
            print('Using expression center')
            expression = torch.tensor(loadmat(self.model.expression_path)[
                                      'expression_center'])[0]

        # Load DNet, model (LNet, ENet)
        D_Net, model = load_model(self.model, self.device)

        stabilized_path = os.path.join(self.model.tmp_dir, 'stabilized.npy')
        if not os.path.isfile(stabilized_path) or self.model.re_preprocess:
            imgs = []
            for idx in tqdm(range(len(frames_pil)), desc='[Step 3] Stabilizing expression in video'):
                if self.model.one_shot:
                    source_img = trans_image(
                        frames_pil[0]).unsqueeze(0).to(self.device)
                    semantic_source_numpy = semantic_npy[0:1]
                else:
                    source_img = trans_image(
                        frames_pil[idx]).unsqueeze(0).to(self.device)
                    semantic_source_numpy = semantic_npy[idx:idx+1]
                ratio = find_crop_norm_ratio(
                    semantic_source_numpy, semantic_npy)
                coeff = transform_semantic(
                    semantic_npy, idx, ratio).unsqueeze(0).to(self.device)

                coeff[:, :64, :] = expression[None, :64, None].to(self.device)
                with torch.no_grad():
                    output = D_Net(source_img, coeff)

                img_stabilized = np.uint8((output['fake_image'].squeeze(0).permute(
                    1, 2, 0).cpu().clamp_(-1, 1).numpy() + 1) / 2. * 255)
                imgs.append(cv2.cvtColor(img_stabilized, cv2.COLOR_RGB2BGR))
            np.save(stabilized_path, imgs)
            del D_Net
        else:
            print('[Step 3] Using saved stabilized video...')
            imgs = np.load(stabilized_path)
        torch.cuda.empty_cache()

        # No need because step 1 (see tools/ffmpeg.py extract_audio) ensures audio is wav
        # if not self.input_audio.endswith('.wav'):
        #     command = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(
        #         self.input_audio, 'temp/{}/temp.wav'.format(self.model.tmp_dir))
        #     subprocess.call(command, shell=True)
        #     self.input_audio = 'temp/{}/temp.wav'.format(self.model.tmp_dir)

        wav = load_wav(self.input_audio, 16000)
        mel = melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan. Add a small epsilon noise to wav file if using TTS voice.')

        mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1

        print(f'[Step 4] Loading audio - {len(mel_chunks)} chunks')
        imgs = imgs[:len(mel_chunks)]
        full_frames = full_frames[:len(mel_chunks)]
        lm = lm[:len(mel_chunks)]

        imgs_enhanced = []
        for idx in tqdm(range(len(imgs)), desc='[Step 5] Enhancing reference frames'):
            img = imgs[idx]
            pred, _, _ = enhancer.process(
                img, img, face_enhance=True, possion_blending=False)
            imgs_enhanced.append(pred)
        gen = self.datagen(imgs_enhanced.copy(), mel_chunks,
                           full_frames, None, (oy1, oy2, ox1, ox2))

        frame_h, frame_w = full_frames[0].shape[:-1]
        out = cv2.VideoWriter(os.path.join(self.model.tmp_dir, 'result.mp4'),
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        if self.model.up_face != 'original':
            instance = GANimationModel()
            instance.initialize()
            instance.setup()

        kp_extractor = KeypointExtractor()
        for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip synthesis', total=int(np.ceil(float(len(mel_chunks)) / self.model.LNet_batch_size)))):
            img_batch = torch.FloatTensor(np.transpose(
                img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(
                mel_batch, (0, 3, 1, 2))).to(self.device)
            img_original = torch.FloatTensor(np.transpose(
                img_original, (0, 3, 1, 2))).to(self.device)/255.  # BGR -> RGB

            with torch.no_grad():
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                pred, low_res = model(mel_batch, img_batch, reference)
                pred = torch.clamp(pred, 0, 1)

                if self.model.up_face in ['sad', 'angry', 'surprise']:
                    tar_aus = exp_aus_dict[self.model.up_face]
                else:
                    pass

                if self.model.up_face == 'original':
                    cur_gen_faces = img_original
                else:
                    test_batch = {'src_img': F.interpolate((img_original * 2 - 1),
                                                           size=(128, 128),
                                                           mode='bilinear'),
                                  'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                    instance.feed_batch(test_batch)
                    instance.forward()
                    cur_gen_faces = F.interpolate(instance.fake_img / 2. + 0.5,
                                                  size=(384, 384),
                                                  mode='bilinear')

                if self.model.without_rl1:
                    incomplete, reference = torch.split(img_batch, 3, dim=1)
                    mask = torch.where(incomplete == 0,
                                       torch.ones_like(incomplete),
                                       torch.zeros_like(incomplete))
                    pred = pred * mask + cur_gen_faces * (1 - mask)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            torch.cuda.empty_cache()

            for p, f, xf, c in zip(pred, frames, f_frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                ff = xf.copy()
                ff[y1:y2, x1:x2] = p

                # mouth region enhancement by GFPGAN
                cropped_faces, restored_faces, restored_img = restorer.enhance(
                    ff, has_aligned=False, only_center_face=True, paste_back=True
                )
                mm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      255, 255, 255, 0, 0, 0, 0, 0, 0]
                mouse_mask = np.zeros_like(restored_img)
                tmp_mask = enhancer.faceparser.process(
                    restored_img[y1:y2, x1:x2], mm)[0]
                mouse_mask[y1:y2, x1:x2] = cv2.resize(
                    tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

                height, width = ff.shape[:2]
                restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in (
                    restored_img, ff, np.float32(mouse_mask))]
                img = Laplacian_Pyramid_Blending_with_mask(
                    restored_img, ff, full_mask[:, :, 0], 10)
                pp = np.uint8(cv2.resize(
                    np.clip(img, 0, 255), (width, height)))

                pp, orig_faces, enhanced_faces = enhancer.process(
                    pp, xf, bbox=c, face_enhance=False, possion_blending=True)
                out.write(pp)

        out.release()

        outfile = add_audio_to_video(self.input_audio, os.path.join(
            self.model.tmp_dir, 'result.mp4'), self.output_path)

        print(
            f'[Final] Inference completed. Output video saved at {outfile}')
