import subprocess


def extract_audio(video_file_path, output_audio_file_path):
    command = ['ffmpeg', '-v', 'quiet', '-i', video_file_path, '-vn', '-acodec',
               'pcm_s16le', '-ar', '44100', '-ac', '2', output_audio_file_path]

    subprocess.run(command)
