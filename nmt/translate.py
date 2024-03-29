import os
import argparse

import ctranslate2
import sentencepiece as spm
import torch


class Translator:

    def __init__(self, input_file, output_file, model_name, device=None):
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.model_path = self.get_model_path()
        self.model = ctranslate2.Translator(
            self.model_path, device=self.device)
        self.sp = spm.SentencePieceProcessor(f'{self.model_path}/source.spm')

    def get_model_path(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, 'models', self.model_name)
        required_files = ['model.bin', 'config.json',
                          'shared_vocabulary.json', 'source.spm', 'target.spm']

        if not all(os.path.exists(os.path.join(model_path, file)) for file in required_files):
            raise Exception(
                'One or more model files are missing. Please check the models directory.')

        return model_path

    def translate_text(self, input_text):
        try:
            input_tokens = self.sp.encode(input_text, out_type=str)
            results = self.model.translate_batch([input_tokens])
            output_tokens = results[0].hypotheses[0]
            return self.sp.decode(output_tokens)
        except Exception as e:
            print(f'Error translating text: {e}')
            return None

    def translate(self):
        if not os.path.exists(self.input_file):
            raise Exception(f'Input file {self.input_file} does not exist.')
        try:
            with open(self.input_file, 'r', encoding='utf-8') as i, open(self.output_file, 'w', encoding='utf-8') as o:
                for line in i:
                    translated_line = self.translate_text(line.strip())
                    if translated_line is not None:
                        o.write(translated_line + '\n')
            return self.output_file
        except Exception as e:
            print(f'Error translating file: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to the source text file to translate')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='Path to the translated output text file')
    parser.add_argument('-m', '--model_name', type=str, required=True, choices=['en-zh', 'zh-en'],
                        help='Translation model used (en-zh or zh-en)')

    args = parser.parse_args()

    translator = Translator(args.input_file, args.output_file, args.model_name)
    output_file_path = translator.translate()

    print(f'Translation complete - files written to {output_file_path}')
