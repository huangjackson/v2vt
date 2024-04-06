import argparse

from termcolor import cprint

from tools.utils import check_dependencies, check_models_and_download
from tools.ffmpeg import extract_audio

from tools.vr.uvr import UltimateVocalRemover
from tools.asr.slice import Slicer
from tools.asr.transcribe import Transcriber
from tools.nmt.translate import Translator

from tts.prepare_datasets.get_text import GetText
from tts.prepare_datasets.get_hubert_wav32k import GetHubertWav32k
from tts.prepare_datasets.get_semantic import GetSemantic
from tts.s2_train import S2Train
from tts.s1_train import S1Train
from tts.inference import TTSInference

from lipsync.inference import LipSyncInference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Path to input video')
    parser.add_argument('-o', '--output', type=str,
                        required=True, help='Path to translated video')

    args = parser.parse_args()

    check_dependencies()
    check_models_and_download(ask=True)

    cprint('[Step 1]: Extracting audio from video...',
           'black', 'on_cyan', attrs=['bold'])
    extract_audio(video_file_path=args.input,
                  output_audio_file_path='logs/audio.wav')

    cprint('[Step 2]: Removing vocals...', 'black', 'on_cyan', attrs=['bold'])
    uvr = UltimateVocalRemover(input_file='logs/audio.wav',
                               output_folder='logs').run()

    cprint('[Step 3]: Slicing audio...', 'black', 'on_cyan', attrs=['bold'])
    slicer = Slicer(input_file='logs/audio_Vocals.wav',
                    output_folder='logs/sliced_audio').run()

    cprint('[Step 4]: Transcribing audio...',
           'black', 'on_cyan', attrs=['bold'])
    transcriber = Transcriber(input_folder='logs/sliced_audio',
                              output_folder='logs').run()

    cprint('[Step 5]: Translating..', 'black', 'on_cyan', attrs=['bold'])
    # TODO: Temporarily hard-coded for zh-en translation
    translator = Translator(input_file='logs/transcript.txt',
                            output_file='logs/translation.txt',
                            model_name='zh-en').run()

    cprint('[Step 6]: Executing TTS...', 'black', 'on_cyan', attrs=['bold'])
    # TODO: Temporarily hard-coded output speech
    tts_1a = GetText().execute()
    tts_1b = GetHubertWav32k().execute()
    tts_1c = GetSemantic().execute()
    tts_2a = S2Train(batch_size=4, total_epoch=8, text_low_lr_rate=0.4,
                     if_save_latest=True, if_save_every_weights=True,
                     save_every_epoch=4).run()
    tts_2b = S1Train(batch_size=4, total_epoch=15, if_dpo=False,
                     if_save_latest=True, if_save_every_weights=True,
                     save_every_epoch=5).run()
    tts_3 = TTSInference(ref_wav_path='logs/sliced_audio/audio_Vocals_1.wav',
                         ref_text='',
                         ref_text_language='zh',
                         text='Good morning everyone. The bible teaches us to be kind and forgiving.',
                         text_language='en',
                         ref_free=True).run()

    cprint('[Step 7]: Executing lip sync...',
           'black', 'on_cyan', attrs=['bold'])
    lipsync = LipSyncInference(input_video=args.input,
                               input_audio='logs/output.wav',
                               output_path=args.output).run()

    cprint(
        f'[Final] Done. File written to {args.output}', 'black', 'on_green', attrs=['bold'])
