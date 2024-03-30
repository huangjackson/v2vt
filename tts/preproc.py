from .config import ModelData
from .prepare_datasets.get_text import GetText
from .prepare_datasets.get_hubert_wav32k import GetHubertWav32k

model = ModelData()

# Temporarily hard-coded paths
step1a = GetText(transcribed_file='TEMP/transcribed.list',
                 sliced_audio_folder='TEMP/sliced_audio',
                 output_folder='TEMP/1a-gettext',
                 roberta_path=model.roberta_path).execute()

step1b = GetHubertWav32k(transcribed_file='TEMP/transcribed.list',
                         sliced_audio_folder='TEMP/sliced_audio',
                         output_folder='TEMP/1b-gethubertwav32k',
                         hubert_path=model.hubert_path).execute()
