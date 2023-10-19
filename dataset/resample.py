import argparse
import glob
import os
from argparse import RawTextHelpFormatter
from multiprocessing import Pool
from shutil import copytree
from tqdm import tqdm
from pydub import AudioSegment

# https://github.com/jaywalnut310/vits/issues/132
# resample should use ffmpeg or sox.
# if there are some blank audio at the begining or end, use librosa to trim it
def resample_file(func_args):
    filename, output_sr, file_ext = func_args
    audio = AudioSegment.from_file(filename, file_ext)
    audio.export(filename+".wav", format="wav", bitrate=output_sr)

def resample_files(input_dir, output_sr, output_dir=None, file_ext="wav", n_jobs=10):
    """
    change all the files to output_sr. sr = sample rate
    """
    if output_dir:
        print("Recursively copying the input folder...")
        copytree(input_dir, output_dir)
        input_dir = output_dir

    print("Resampling the audio files...")
    audio_files = glob.glob(os.path.join(input_dir, f"**/*.{file_ext}"), recursive=True)
    print(f"Found {len(audio_files)} files...")
    audio_files = list(zip(audio_files, len(audio_files) * [output_sr], [file_ext] * len(audio_files)))
    with Pool(processes=n_jobs) as p:
        with tqdm(total=len(audio_files)) as pbar:
            for _, _ in enumerate(p.imap_unordered(resample_file, audio_files)):
                pbar.update()

    print("Done ! removing original file if needed")
    if file_ext != "wav":
        for filename, _, _ in audio_files:
            os.remove(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Resample a folder recusively with librosa
                       Can be used in place or create a copy of the folder as an output.\n\n
                       Example run:
                            python TTS/bin/resample.py
                                --input_dir /root/LJSpeech-1.1/
                                --output_sr 22050
                                --output_dir /root/resampled_LJSpeech-1.1/
                                --file_ext wav
                                --n_jobs 24
                    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        required=True,
        help="Path of the folder containing the audio files to resample",
    )

    parser.add_argument(
        "--output_sr",
        type=int,
        default=22050,
        required=False,
        help="Samlple rate to which the audio files should be resampled",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=False,
        help="Path of the destination folder. If not defined, the operation is done in place",
    )

    parser.add_argument(
        "--file_ext",
        type=str,
        default="wav",
        required=False,
        help="Extension of the audio files to resample",
    )

    parser.add_argument(
        "--n_jobs", type=int, default=None, help="Number of threads to use, by default it uses all cores"
    )

    args = parser.parse_args()

    resample_files(args.input_dir, args.output_sr, args.output_dir, args.file_ext, args.n_jobs)
