from glob import glob
import argparse
import os
from multiprocessing import Pool
from shutil import copytree
from tqdm import tqdm
from dataset.dataset_util import download_kaggle_dataset, extract_archive, resample_file
from typing import Optional

def download_vctk(save_path: str, use_kaggle: Optional[bool] = False):
    """Download and extract VCTK dataset.
    Args:
        save_path (str): path to the directory where the dataset will be stored.
        use_kaggle (bool, optional): Downloads vctk dataset from kaggle. Is generally faster. Defaults to False.
    """
    if use_kaggle:
        download_kaggle_dataset("mfekadu/english-multispeaker-corpus-for-voice-cloning", "VCTK", save_path)
    else:
        os.makedirs(save_path, exist_ok=True)
        url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
        # download_url(url, save_path)
        basename = os.path.basename(url)
        archive = os.path.join(save_path, basename)
        print(" > Extracting archive file...")
        extract_archive(archive)


def load_vctk_metas(root_path:str, wavs_path="wav48_silence_trimmed", mic="mic1", ignored_speakers=None):
    """
    load VCTK dataset file meta
    """
    file_ext = "flac"
    items = []
    meta_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, "r", encoding="utf-8") as file_text:
            text = file_text.readlines()[0]
        # p280 has no mic2 recordings
        if speaker_id == "p280":
            wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + f"_mic1.{file_ext}")
        else:
            wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + f"_{mic}.{file_ext}")
        if os.path.exists(wav_file):
            items.append({"text": text, "audio": wav_file, "speaker": "VCTK_" + speaker_id, "root_path": root_path, "language":"en"})
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
    parser = argparse.ArgumentParser(description="download and resample VCTK dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--current_path", type=str, default=None, required=False, help="Path of the folder containing the audio files to resample", )
    parser.add_argument("--sample_rate", type=int, default=22050, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--resample_threads", type=int, default=10, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    # VCTK_DOWNLOAD_PATH = os.path.join(args.current_path, "VCTK")
    VCTK_DOWNLOAD_PATH = "D:\\dataset\\VCTK"

    print(">>> Downloading VCTK dataset:")
    download_vctk(VCTK_DOWNLOAD_PATH)
    print(">>> resampling VCTK dataset:")
    resample_files(VCTK_DOWNLOAD_PATH, args.sample_rate, file_ext="flac", n_jobs=args.resample_threads)

