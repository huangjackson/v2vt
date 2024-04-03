import os
import re
from glob import glob

import torch
from faster_whisper import WhisperModel

# Required to prevent error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def natsort(s): return [int(t) if t.isdigit()
                        else t.lower() for t in re.split('(\d+)', s)]


class Transcriber:

    def __init__(self, input_folder, output_folder, model_size='large-v3', language='auto', precision='float16'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_size = model_size
        self.language = None if language == 'auto' else language
        self.precision = precision
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.precision)
        except Exception as e:
            raise Exception(
                f'An error occured while loading the faster-whisper model: {e}')

    def transcribe(self):
        dataset = []

        dataset_file_name = os.path.basename(self.input_folder)
        dataset_file_path = os.path.join(
            self.output_folder, 'transcribed.list')

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for file in sorted(glob(os.path.join(self.input_folder, '**/*.wav'), recursive=True), key=natsort):
            try:
                segments, info = self.model.transcribe(
                    audio=file,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=700),
                    language=self.language)

                text = ''.join([segment.text for segment in segments])

                dataset.append(
                    f'{file}|{dataset_file_name}|{info.language.upper()}|{text}')
            except Exception as e:
                return print(f'An error occurred during transcription: {e}')

        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dataset))

        return self.output_folder
