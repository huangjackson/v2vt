import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    required=True, help='Path to input video')
parser.add_argument('-o', '--output', type=str,
                    required=True, help='Path to output file')

args = parser.parse_args()

check_dependencies()
check_models_and_install()

# TODO: 1) extract audio from video
#       2) uvr on extracted audio
#       3) slice
#       4) transcribe & create dataset
#       5) translate
#       6) tts
#       7) lip sync
