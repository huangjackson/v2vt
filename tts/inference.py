import os
import re
from time import time as ttime

import torch
import numpy as np
import librosa
import LangSegment
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.io import wavfile

from .config import ModelData
from .feature_extractor import cnhubert
from .module.models import SynthesizerTrn
from .module.mel_processing import spectrogram_torch
from .AR.models.t2s_lightning_module import Text2SemanticLightningModule
from .text.cleaner import clean_text
from .text import cleaned_text_to_sequence

# TODO: Absolute import unlike others, make every import absolute?
from tools.ffmpeg import load_audio


splits = {'，', '。', '？', '！', ',', '.', '?', '!', '~', ':', '：', '—', '…', }


def get_first(text):
    pattern = '[' + ''.join(re.escape(sep) for sep in splits) + ']'
    text = re.split(pattern, text)[0].strip()
    return text


def split(text):
    text = text.replace('……', '。').replace('——', '，')
    if text[-1] not in splits:
        text += '。'
    i_split_head = i_split_tail = 0
    len_text = len(text)
    texts = []
    while i_split_head < len_text:
        if text[i_split_head] in splits:
            texts.append(text[i_split_tail:i_split_head])
            i_split_tail = i_split_head + 1
        i_split_head += 1
    return texts


def cut1(text):
    # Split every 4 sentences
    text = text.strip('\n')
    texts = split(text)
    split_idx = list(range(0, len(texts), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append(''.join(texts[split_idx[idx]:split_idx[idx + 1]]))
    else:
        opts = [text]
    return '\n'.join(opts)


class DictToAttrRecursive(dict):

    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f'Attribute {item} not found')

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f'Attribute {item} not found')


class TTSInference:

    def __init__(self, output_folder, ref_wav_path, ref_text, ref_text_language, text,
                 text_language, top_k=5, top_p=1, temperature=1, ref_free=True):
        self.output_folder = output_folder
        self.ref_wav_path = ref_wav_path
        self.ref_text = ref_text
        self.ref_text_language = ref_text_language  # 'en', 'zh', 'all_zh', 'auto'
        self.text = text
        self.text_language = text_language  # 'en', 'zh', 'all_zh', 'auto'
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.ref_free = ref_free if (
            ref_text is not None and len(ref_text) != 0) else True

        self.model = ModelData()

        self.sovits_path = None
        self.gpt_path = None
        self.tokenizer = None
        self.bert_model = None
        self.ssl_model = None
        self.hps = None
        self.vq_model = None
        self.hz = None
        self.max_sec = None
        self.t2s_model = None
        self.config = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32

        cnhubert.cnhubert_base_path = self.model.hubert_path

        # Load models
        try:
            self.sovits_path = self.find_latest_sovits_weights()
            self.gpt_path = self.find_latest_gpt_weights()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.roberta_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(
                self.model.roberta_path).to(self.device)
            self.ssl_model = cnhubert.get_model().to(self.device)
            if self.sovits_path is not None and self.gpt_path is not None:
                self.load_sovits()
                self.load_gpt()
            else:
                raise Exception('No models found')
        except Exception as e:
            raise Exception(f'Error while loading models: {e}')

    def find_latest_sovits_weights(self):
        weights = os.listdir(self.model.sovits_weights_path)
        weights = [os.path.join(self.model.sovits_weights_path, weight)
                   for weight in weights if weight.endswith('.pth')]
        weights.sort(key=os.path.getmtime)
        return weights[-1]

    def find_latest_gpt_weights(self):
        weights = os.listdir(self.model.gpt_weights_path)
        weights = [os.path.join(self.model.gpt_weights_path, weight)
                   for weight in weights if weight.endswith('.ckpt')]
        weights.sort(key=os.path.getmtime)
        return weights[-1]

    def load_sovits(self):
        dict_s2 = torch.load(self.sovits_path, map_location='cpu')

        self.hps = DictToAttrRecursive(dict_s2['config'])
        self.hps.model.semantic_frame_rate = '25hz'

        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(self.device)

        # TODO: detect if pretrained - if not, do following:
        del self.vq_model.enc_q

        self.vq_model.eval()
        print(self.vq_model.load_state_dict(dict_s2['weight'], strict=False))

    def load_gpt(self):
        self.hz = 50
        dict_s1 = torch.load(self.gpt_path, map_location='cpu')

        self.config = dict_s1['config']
        self.max_sec = self.config['data']['max_sec']

        self.t2s_model = Text2SemanticLightningModule(
            self.config, '****', is_train=False)
        self.t2s_model.load_state_dict(dict_s1['weight'])
        self.t2s_model = self.t2s_model.to(self.device)

        self.t2s_model.eval()
        total = sum([param.nelement()
                    for param in self.t2s_model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))

    def get_spepc(self, filename):
        audio = load_audio(filename, int(self.hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return spec

    def clean_text_inf(self, text, language):
        phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

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

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace('all_', '')
        if language == 'zh':
            bert = self.get_bert_feature(norm_text, word2ph).to(
                self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                torch.float32,
            ).to(self.device)

        return bert

    def get_phones_and_bert(self, text, language):
        if language in {'en', 'all_zh'}:
            language = language.replace('all_', '')
            if language == 'en':
                LangSegment.setfilters(['en'])
                formattext = ' '.join(tmp['text']
                                      for tmp in LangSegment.getTexts(text))
            else:
                formattext = text
            while '  ' in formattext:
                formattext = formattext.replace('  ', ' ')
            phones, word2ph, norm_text = self.clean_text_inf(
                formattext, language)
            if language == 'zh':
                bert = self.get_bert_feature(
                    norm_text, word2ph).to(self.device)
            else:
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float32,
                ).to(self.device)
        elif language in {'zh', 'auto'}:
            textlist = []
            langlist = []
            LangSegment.setfilters(['zh', 'ja', 'en', 'ko'])
            if language == 'auto':
                for tmp in LangSegment.getTexts(text):
                    if tmp['lang'] == 'ko':
                        langlist.append('zh')
                        textlist.append(tmp['text'])
                    else:
                        langlist.append(tmp['lang'])
                        textlist.append(tmp['text'])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp['lang'] == 'en':
                        langlist.append(tmp['lang'])
                    else:
                        langlist.append(language)
                    textlist.append(tmp['text'])

            print(textlist)
            print(langlist)

            phones_list = []
            bert_list = []
            norm_text_list = []

            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(
                    textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)

            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ' '.join(norm_text_list)

        return phones, bert.to(self.dtype), norm_text

    def merge_short_text_in_array(self, texts, threshold):
        if (len(texts)) < 2:
            return texts
        result = []
        text = ''
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ''
        if (len(text) > 0):
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    def run(self):
        t0 = ttime()

        if not self.ref_free:
            self.ref_text = self.ref_text.strip('\n')
            if self.ref_text[-1] not in splits:
                self.ref_text += '。' if self.ref_text_language == 'zh' else '.'

        self.text = self.text.strip('\n')
        if self.text[0] not in splits and len(get_first(self.text)) < 4:
            self.text = '。' + self.text if self.text_language == 'zh' else '.' + self.text

        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float32,
        )

        with torch.no_grad():
            wav16k, sr = librosa.load(self.ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError(
                    'Reference wav file is outside the range of 3-10 seconds')
            wav16k = torch.from_numpy(wav16k).to(self.device)
            zero_wav_torch = torch.from_numpy(zero_wav).to(self.device)

            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                'last_hidden_state'].transpose(1, 2)
            codes = self.vq_model.extract_latent(ssl_content)

            prompt_semantic = codes[0, 0]

        t1 = ttime()

        self.text = cut1(self.text)
        # TODO: other split methods

        while '\n\n' in self.text:
            self.text = self.text.replace('\n\n', '\n')
        texts = self.text.split('\n')
        texts = self.merge_short_text_in_array(texts, 5)

        audio_opt = []
        if not self.ref_free:
            phones1, bert1, norm_text1 = self.get_phones_and_bert(
                self.ref_text, self.ref_text_language)

        for text in texts:
            if len(text.strip()) == 0:
                continue
            if text[-1] not in splits:
                text += '。' if self.text_language == 'zh' else '.'
            phones2, bert2, norm_text2 = self.get_phones_and_bert(
                text, self.text_language)

            if not self.ref_free:
                bert = torch.cat([bert1, bert2], 1).to(
                    self.device).unsqueeze(0)
                all_phoneme_ids = torch.LongTensor(
                    phones1 + phones2).to(self.device).unsqueeze(0)
            else:
                bert = bert2.to(self.device).unsqueeze(0)
                all_phoneme_ids = torch.LongTensor(
                    phones2).to(self.device).unsqueeze(0)

            all_phoneme_len = torch.tensor(
                [all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

            t2 = ttime()

            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if self.ref_free else prompt,
                    bert,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    early_stop_num=self.hz * self.max_sec,
                )

            t3 = ttime()

            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refer = self.get_spepc(self.ref_wav_path).to(self.device)

            audio = (
                self.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                    refer,
                ).detach().cpu().numpy()[0, 0]
            )

            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio = audio / max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)

            t4 = ttime()
        print('%.3f\t%.3f\t%.3f\t%.3f' % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))

        audio_data = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
        output_file = 'output.wav'
        wavfile.write(os.path.join(self.output_folder, output_file),
                      self.hps.data.sampling_rate, audio_data)
