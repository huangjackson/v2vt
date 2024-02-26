# Includes modified code from https://github.com/RVC-Boss/GPT-SoVITS

"""
MIT License

Copyright (c) 2024 RVC-Boss

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import argparse
import traceback
import glob

from faster_whisper import WhisperModel

# Required to prevent error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

language_code_list = [
    'af', 'am', 'ar', 'as', 'az',
    'ba', 'be', 'bg', 'bn', 'bo',
    'br', 'bs', 'ca', 'cs', 'cy',
    'da', 'de', 'el', 'en', 'es',
    'et', 'eu', 'fa', 'fi', 'fo',
    'fr', 'gl', 'gu', 'ha', 'haw',
    'he', 'hi', 'hr', 'ht', 'hu',
    'hy', 'id', 'is', 'it', 'ja',
    'jw', 'ka', 'kk', 'km', 'kn',
    'ko', 'la', 'lb', 'ln', 'lo',
    'lt', 'lv', 'mg', 'mi', 'mk',
    'ml', 'mn', 'mr', 'ms', 'mt',
    'my', 'ne', 'nl', 'nn', 'no',
    'oc', 'pa', 'pl', 'ps', 'pt',
    'ro', 'ru', 'sa', 'sd', 'si',
    'sk', 'sl', 'sn', 'so', 'sq',
    'sr', 'su', 'sv', 'sw', 'ta',
    'te', 'tg', 'th', 'tk', 'tl',
    'tr', 'tt', 'uk', 'ur', 'uz',
    'vi', 'yi', 'yo', 'zh', 'yue',
    'auto']


def execute_asr(input_folder, output_folder, model_size, language, precision):
    if language == 'auto':
        language = None

    try:
        model = WhisperModel(model_size, device='cuda', compute_type=precision)
    except:
        return print(traceback.format_exc())

    output = []
    output_file_name = os.path.basename(input_folder)
    output_file_path = os.path.abspath(
        f'{output_folder}/{output_file_name}.list')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in glob.glob(os.path.join(input_folder, '**/*.wav'), recursive=True):
        try:
            segments, info = model.transcribe(
                audio=file,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language)

            text = ''

            for segment in segments:
                text += segment.text

            output.append(
                f'{file}|{output_file_name}|{info.language.upper()}|{text}')
        except:
            return print(traceback.format_exc())

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    return output_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='Path to the source folder containing WAV files.')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Output folder to store transcriptions.')
    parser.add_argument('-s', '--model_size', type=str, default='large-v3',
                        help='Model size of faster-whisper.')
    parser.add_argument('-l', '--language', type=str, default='auto',
                        choices=language_code_list,
                        help='Language of the source audio files.')
    parser.add_argument('-p', '--precision', type=str, default='float16', choices=['float16', 'float32'],
                        help='fp16 or fp32')

    cmd = parser.parse_args()

    output_file_path = execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_size=cmd.model_size,
        language=cmd.language,
        precision=cmd.precision,
    )

    print(f'ASR complete - files written to {output_file_path}')
