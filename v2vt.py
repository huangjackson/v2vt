import argparse

from tools.utils import (
    check_dependencies, check_models_and_install
)
from tools.ffmpeg import (
    extract_audio
)

from tools.vr.uvr import UltimateVocalRemover
from tools.asr.slice import Slicer
from tools.asr.transcribe import Transcriber
from tools.nmt.translate import Translator

from tts.config import TTSModel
from tts.prepare_datasets.get_text import GetText
from tts.prepare_datasets.get_hubert_wav32k import GetHubertWav32k
from tts.prepare_datasets.get_semantic import GetSemantic
from tts.s2_train import S2Train
from tts.s1_train import S1Train
from tts.inference import TTSInference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Path to input video')
    parser.add_argument('-o', '--output', type=str,
                        required=True, help='Path to translated video')

    args = parser.parse_args()

    check_dependencies()
    check_models_and_install()

    # model = TTSModel()

    # 1: Preprocessing
    # step1a = GetText(transcribed_file='logs/transcribed.list',
    #                  sliced_audio_folder='logs/sliced_audio',
    #                  output_folder='logs/1-preproc',
    #                  roberta_path=model.roberta_path).execute()
    # step1b = GetHubertWav32k(transcribed_file='logs/transcribed.list',
    #                          sliced_audio_folder='logs/sliced_audio',
    #                          output_folder='logs/1-preproc',
    #                          hubert_path=model.hubert_path).execute()
    # step1c = GetSemantic(transcribed_file='logs/transcribed.list',
    #                      output_folder='logs/1-preproc',
    #                      pretrained_s2G_path=model.pretrained_s2G_path,
    #                      s2_config_path=model.s2_config_path).execute()
    #
    # 2: Training
    # step2a = S2Train(batch_size=4, total_epoch=8, text_low_lr_rate=0.4,
    #                  if_save_latest=True, if_save_every_weights=True,
    #                  save_every_epoch=4).run()
    # step2b = S1Train(batch_size=4, total_epoch=15, if_dpo=False,
    #                  if_save_latest=True, if_save_every_weights=True,
    #                  save_every_epoch=5).run()
    #
    # 3: Inference
    # step3 = TTSInference(output_folder='logs',
    #                      ref_wav_path='logs/sliced_audio/audio_Vocals_1.wav',
    #                      ref_text='',
    #                      ref_text_language='zh',
    #                      text='Good morning everyone. The bible teaches us to be kind and forgiving.',
    #                      text_language='en',
    #                      ref_free=True).run()

    # TODO: 1) extract audio from video
    #       2) uvr on extracted audio
    #       3) slice
    #       4) transcribe & create dataset
    #       5) translate
    #       6) tts
    #       7) lip sync
