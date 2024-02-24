# Includes modified code from https://github.com/Anjok07/ultimatevocalremovergui

import numpy
import librosa
from onnx import load
import onnxruntime
from onnx2pytorch import ConvertModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uvr import ModelData


def prepare_mix(mix):
    if not isinstance(mix, numpy.ndarray):
        mix = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if mix.ndim == 1:
        mix = numpy.asfortranarray([mix, mix])

    return mix


class SeparateAttributes:
    def __init__(self, model_data: ModelData, process_data: dict):
        self.cached_source_callback = process_data['cached_source_callback']
        self.model_basename = model_data.model_basename
        self.primary_sources = self.cached_source_callback(
            model_name=self.model_basename)


class SeparateMDX(SeparateAttributes):
    def separate(self):
        samplerate = 44100

        if self.mdx_segment_size == self.dim_t and not self.is_other_gpu:
            ort_ = onnxruntime.InferenceSession(
                self.model_path, providers=self.run_type)
            self.model_run = lambda spek: ort_.run(
                None, {'input': spek.cpu().numpy()})[0]
        else:
            self.model_run = ConvertModel(load(self.model_path))
            self.model_run.to(self.device).eval()

        mix = prepare_mix(self.audio_file)

        # Incomplete -- add rest of separation code
        
