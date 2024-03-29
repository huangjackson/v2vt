from .config import ModelData
from .prepare_datasets.get_text import GetText

model = ModelData()

# temporarily hard-coded paths
step1a = GetText(transcribed_file='TEMP/transcribed.list',
                 sliced_audio_folder='TEMP/sliced_audio',
                 output_folder='TEMP/1a-gettext',
                 roberta_path=model.roberta_path).execute()
