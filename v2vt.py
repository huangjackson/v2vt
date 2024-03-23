import sys
import argparse
import tempfile

from tools.utils import (
    check_dependencies, check_models_and_install
)
from tools.ffmpeg import (
    extract_audio
)

from vr.uvr import UltimateVocalRemover
from asr.slice import Slicer
from asr.transcribe import Transcriber
from nmt.translate import Translator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Path to input video')
    parser.add_argument('-o', '--output', type=str,
                        required=True, help='Path to output file')

    args = parser.parse_args()

    check_dependencies()
    check_models_and_install()

    try:
        # TODO: Replace args.output with tempfile, args.output should be final output (translated video)
        extract_audio(args.input, args.output)
    except Exception as e:
        print(
            f'An error occured while extracting audio from video via FFmpeg: {e}')
        sys.exit(1)

    # TODO: 1) uvr on extracted audio
    #       2) slice
    #       3) transcribe & create dataset
    #       4) translate
    #       5) tts
    #       6) lip sync
