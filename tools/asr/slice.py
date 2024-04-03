import os

from pydub import AudioSegment
from pydub.silence import split_on_silence


class Slicer:

    def __init__(self, input_file, output_folder, min_length=5000, min_silence=500, silence_thresh=-16, silence_keep=250):
        self.input_file = input_file
        self.output_folder = output_folder
        self.min_length = min_length
        self.min_silence = min_silence
        self.silence_thresh = silence_thresh
        self.silence_keep = silence_keep
        try:
            self.audio = AudioSegment.from_wav(self.input_file)
        except Exception as e:
            raise Exception(
                f'An error occured while loading the audio file: {e}')

    def slice(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        try:
            audio_chunks = split_on_silence(
                self.audio,
                min_silence_len=self.min_silence,
                silence_thresh=self.audio.dBFS + self.silence_thresh,
                keep_silence=self.silence_keep
            )
        except Exception as e:
            return print(f'An error occured during audio processing: {e}')

        if not audio_chunks:
            return print(
                'No audio chunks were detected. Check silence parameters.')

        output_chunks = [audio_chunks[0]]

        for chunk in audio_chunks[1:]:
            if len(output_chunks[-1]) < self.min_length:
                output_chunks[-1] += chunk
            else:
                output_chunks.append(chunk)

        output_file_name = os.path.splitext(
            os.path.basename(self.input_file))[0]

        for i, chunk in enumerate(output_chunks):
            try:
                chunk.export(os.path.join(self.output_folder,
                                          f'{output_file_name}_{i}.wav'))
                print(f'Exported {output_file_name}_{i}.wav')
            except Exception as e:
                return print(f'Failed to export {output_file_name}_{i}.wav: {e}')

        return self.output_folder
