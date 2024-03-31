# Testing script for TTS pipeline
# Temporarily hard-coded paths & parameters for testing purposes

from .config import ModelData
from .prepare_datasets.get_text import GetText
from .prepare_datasets.get_hubert_wav32k import GetHubertWav32k
from .prepare_datasets.get_semantic import GetSemantic
from .s2_train import S2Train

model = ModelData()

# 1: Preprocessing
step1a = GetText(transcribed_file='logs/transcribed.list',
                 sliced_audio_folder='logs/sliced_audio',
                 output_folder='logs/1-preproc',
                 roberta_path=model.roberta_path).execute()

step1b = GetHubertWav32k(transcribed_file='logs/transcribed.list',
                         sliced_audio_folder='logs/sliced_audio',
                         output_folder='logs/1-preproc',
                         hubert_path=model.hubert_path).execute()

step1c = GetSemantic(transcribed_file='logs/transcribed.list',
                     output_folder='logs/1-preproc',
                     pretrained_s2G_path=model.pretrained_s2G_path,
                     s2_config_path=model.s2_config_path).execute()

# 2: Training
step2a = S2Train(batch_size=4, total_epoch=8, text_low_lr_rate=0.4,
                 if_save_latest=True, if_save_every_weights=True,
                 save_every_epoch=4).run()
