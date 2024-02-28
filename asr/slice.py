import os
import argparse

from pydub import AudioSegment
from pydub.silence import split_on_silence


class AudioSlicer:
    def __init__(self, input_file, output_folder, min_length):
        self.input_file = input_file
        self.output_folder = output_folder
        self.min_length = min_length
        self.audio = AudioSegment.from_wav(self.input_file)

    def slice_audio(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        audio_chunks = split_on_silence(
            self.audio,
            min_silence_len=500,
            silence_thresh=self.audio.dBFS - 16,
            keep_silence=250
        )

        output_chunks = [audio_chunks[0]]

        for chunk in audio_chunks[1:]:
            if len(output_chunks[-1]) < self.min_length:
                output_chunks[-1] += chunk
            else:
                output_chunks.append(chunk)

        output_file_name = os.path.splitext(
            os.path.basename(self.input_file))[0]

        for i, chunk in enumerate(output_chunks):
            chunk.export(os.path.join(self.output_folder,
                         f'{output_file_name}_{i}.wav'))
            print(f'Exported {output_file_name}_{i}.wav')

        return self.output_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True,
                        help='Path to WAV file to slice')
    parser.add_argument('-o', '--output_folder', required=True,
                        help='Output folder to store sliced audio clips')
    parser.add_argument('--min_length', type=int, default=5000,
                        help='Minimum length of sliced audio clips in ms (default = 5000)')

    args = parser.parse_args()

    slicer = AudioSlicer(args.input_file, args.output_folder, args.min_length)
    output_path = slicer.slice_audio()

    print(f'Audio slicing complete -- files written to {output_path}')
