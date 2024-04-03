import os
import time
import face_alignment
import numpy as np
import torch
from tqdm import tqdm


class KeypointExtractor():
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device=device)

    def extract_keypoint(self, images, name=None, info=True):
        if isinstance(images, list):
            keypoints = []
            if info:
                i_range = tqdm(images, desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                current_kp = self.extract_keypoint(image)
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    keypoints = self.detector.get_landmarks_from_image(np.array(images))[
                        0]
                    break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[
                           0]+'.txt', keypoints.reshape(-1))
            return keypoints
