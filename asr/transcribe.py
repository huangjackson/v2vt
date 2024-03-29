# MIT License
#
# Copyright (c) 2024 RVC-Boss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import re
import argparse
from glob import glob

import torch
from faster_whisper import WhisperModel

# Required to prevent error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def natsort(s): return [int(t) if t.isdigit()
                        else t.lower() for t in re.split('(\d+)', s)]


class Transcriber:

    def __init__(self, input_folder, output_folder, model_size, language, precision):
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
            self.output_folder, f'{dataset_file_name}.list')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='Path to folder containing WAV files to transcribe')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Output folder to store transcriptions')
    parser.add_argument('-s', '--model_size', type=str, default='large-v3',
                        help='Model size of faster-whisper')
    parser.add_argument('-l', '--language', type=str, default='auto',
                        choices=['en', 'zh', 'auto'],
                        help='Language of the source audio files')
    parser.add_argument('-p', '--precision', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='Audio precision (16-bit or 32-bit)')

    args = parser.parse_args()

    transcriber = Transcriber(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_size=args.model_size,
        language=args.language,
        precision=args.precision,
    )

    output_path = transcriber.transcribe()

    print(f'ASR complete - files written to {output_path}')
