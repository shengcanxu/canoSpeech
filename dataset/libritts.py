from glob import glob
import argparse
import os
from multiprocessing import Pool
from shutil import copytree
from tqdm import tqdm
from dataset.dataset_util import resample_file

def load_libritts_metas(root_path:str, wavs_path="train-clean-100"):
    items = []
    meta_files = glob(f"{os.path.join(root_path, wavs_path)}/**/*.normalized.txt", recursive=True)

    for meta_file in meta_files:
        _, speaker_id, book_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        with open(meta_file, "r", encoding="utf-8") as file_text:
            text = file_text.readlines()[0]

        wav_file = meta_file.replace(".normalized.txt", ".wav")
        if os.path.exists(wav_file):
            items.append({"text": text, "audio": wav_file, "speaker": "libritts_" + speaker_id, "root_path": root_path, "language":"en"})
        else:
            print(f" [!] wav files don't exist - {wav_file}")
    return items

def resample_files(input_dir, output_sr, output_dir=None, file_ext="wav", n_jobs=10):
    """
    change all the files to output_sr. sr = sample rate
    """
    if output_dir:
        print("Recursively copying the input folder...")
        copytree(input_dir, output_dir)
        input_dir = output_dir

    print("Resampling the audio files...")
    audio_files = glob(os.path.join(input_dir, f"**/*.{file_ext}"), recursive=True)
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
    parser = argparse.ArgumentParser(description="resample LibriTTS dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--sample_rate", type=int, default=22050, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--resample_threads", type=int, default=10, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    # LibriTTS_DOWNLOAD_PATH = "D:/dataset/LibriTTS/train-clean-100"

    # print(">>> resampling LibriTTS dataset:")
    # resample_files(LibriTTS_DOWNLOAD_PATH, args.sample_rate, file_ext="wav", n_jobs=args.resample_threads)

    LibriTTS_DOWNLOAD_PATH = "D:/dataset/LibriTTS"
    items = load_libritts_metas(LibriTTS_DOWNLOAD_PATH)
    print(len(items))
    print(items[0])