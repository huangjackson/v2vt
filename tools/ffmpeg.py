import subprocess

import numpy as np


def extract_audio(video_file_path, output_audio_file_path):
    # Extract audio from video
    command = ['ffmpeg', '-v', 'quiet', '-i', video_file_path, '-vn', '-acodec',
               'pcm_s16le', '-ar', '44100', '-ac', '2', output_audio_file_path]

    try:
        subprocess.run(command)
    except Exception as e:
        raise Exception(f'Error while extracting audio: {e}')


def load_audio(audio_file_path, sr):
    # Decode audio while downmixing and resampling
    command = ['ffmpeg', '-v', 'quiet', '-nostdin', '-threads', '0', '-i', audio_file_path,
               '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '1', '-ar', str(sr), '-']

    try:
        out = subprocess.run(command, capture_output=True, check=True)
        return np.frombuffer(out.stdout, np.float32).flatten()
    except Exception as e:
        raise Exception(f'Error while loading audio: {e}')
