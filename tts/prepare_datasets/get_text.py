# Modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/prepare_datasets/1-get-text.py

import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ..config import TTSModel
from ..text.cleaner import clean_text


class GetText:

    def __init__(self):
        self.model = TTSModel()

        self.txt_path = os.path.join(self.model.preproc_dir, 'phoneme.txt')
        self.bert_dir = os.path.join(self.model.preproc_dir, 'bert')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.roberta_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(
                self.model.roberta_path).to(self.device)
        except Exception as e:
            raise Exception(f'Error while loading roberta model: {e}')

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt')
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res['hidden_states'][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def process(self, data, res):
        for name, text, lan in data:
            try:
                name = os.path.basename(name)
                phones, word2ph, norm_text = clean_text(
                    text.replace('%', '-').replace('￥', ','), lan
                )
                path_bert = f'{self.bert_dir}/{name}.pt'
                if os.path.exists(path_bert) == False and lan == 'zh':
                    bert_feature = self.get_bert_feature(
                        norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    torch.save(bert_feature, path_bert)
                phones = ' '.join(phones)
                res.append([name, phones, word2ph, norm_text])
            except Exception as e:
                print(f'Error while processing {name}: {e}')

    def execute(self):
        os.makedirs(self.model.preproc_dir, exist_ok=True)
        os.makedirs(self.bert_dir, exist_ok=True)

        todo = []
        res = []
        with open(self.model.transcript_path, 'r', encoding='utf8') as f:
            lines = f.read().strip('\n').split('\n')

        for line in lines:
            try:
                wav_name, spk_name, language, text = line.split('|')
                todo.append(
                    [wav_name, text, language.lower()]
                )
            except Exception as e:
                print(f'Error while processing {line}: {e}')

        self.process(todo, res)

        output = []
        for name, phones, word2ph, norm_text in res:
            output.append(f'{name}\t{phones}\t{word2ph}\t{norm_text}')
        with open(self.txt_path, 'w', encoding='utf8') as f:
            f.write('\n'.join(output) + '\n')
