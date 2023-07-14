import argparse
import os
from glob import glob

from dataset.resample import resample_files
from dataset.download_util import download_kaggle_dataset, download_url, extract_archive
from typing import Optional


VCTK_gender_dict = {
        'P225': 'Female',
        'P226': 'Male',
        'P227': 'Male',
        'P228': 'Female',
        'P229': 'Female',
        'P230': 'Female',
        'P231': 'Female',
        'P232': 'Male',
        'P233': 'Female',
        'P234': 'Female',
        'P236': 'Female',
        'P237': 'Male',
        'P238': 'Female',
        'P239': 'Female',
        'P240': 'Female',
        'P241': 'Male',
        'P243': 'Male',
        'P244': 'Female',
        'P245': 'Male',
        'P246': 'Male',
        'P247': 'Male',
        'P248': 'Female',
        'P249': 'Female',
        'P250': 'Female',
        'P251': 'Male',
        'P252': 'Male',
        'P253': 'Female',
        'P254': 'Male',
        'P255': 'Male',
        'P256': 'Male',
        'P257': 'Female',
        'P258': 'Male',
        'P259': 'Male',
        'P260': 'Male',
        'P261': 'Female',
        'P262': 'Female',
        'P263': 'Male',
        'P264': 'Female',
        'P265': 'Female',
        'P266': 'Female',
        'P267': 'Female',
        'P268': 'Female',
        'P269': 'Female',
        'P270': 'Male',
        'P271': 'Male',
        'P272': 'Male',
        'P273': 'Male',
        'P274': 'Male',
        'P275': 'Male',
        'P276': 'Female',
        'P277': 'Female',
        'P278': 'Male',
        'P279': 'Male',
        'P280': 'Female',
        'P281': 'Male',
        'P282': 'Female',
        'P283': 'Male',
        'P284': 'Male',
        'P285': 'Male',
        'P286': 'Male',
        'P287': 'Male',
        'P288': 'Female',
        'P292': 'Male',
        'P293': 'Female',
        'P294': 'Female',
        'P295': 'Female',
        'P297': 'Female',
        'P298': 'Male',
        'P299': 'Female',
        'P300': 'Female',
        'P301': 'Female',
        'P302': 'Male',
        'P303': 'Female',
        'P304': 'Male',
        'P305': 'Female',
        'P306': 'Female',
        'P307': 'Female',
        'P308': 'Female',
        'P310': 'Female',
        'P311': 'Male',
        'P312': 'Female',
        'P313': 'Female',
        'P314': 'Female',
        'P316': 'Male',
        'P317': 'Female',
        'P318': 'Female',
        'P323': 'Female',
        'P326': 'Male',
        'P329': 'Female',
        'P330': 'Female',
        'P333': 'Female',
        'P334': 'Male',
        'P335': 'Female',
        'P336': 'Female',
        'P339': 'Female',
        'P340': 'Female',
        'P341': 'Female',
        'P343': 'Female',
        'P345': 'Male',
        'P347': 'Male',
        'P351': 'Female',
        'P360': 'Male',
        'P361': 'Female',
        'P362': 'Female',
        'P363': 'Male',
        'P364': 'Male',
        'P374': 'Male',
        'P376': 'Male',
        'S5': 'Female',
}

def download_vctk(path: str, use_kaggle: Optional[bool] = False):
    """Download and extract VCTK dataset.

    Args:
        path (str): path to the directory where the dataset will be stored.

        use_kaggle (bool, optional): Downloads vctk dataset from kaggle. Is generally faster. Defaults to False.
    """
    if use_kaggle:
        download_kaggle_dataset("mfekadu/english-multispeaker-corpus-for-voice-cloning", "VCTK", path)
    else:
        os.makedirs(path, exist_ok=True)
        url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
        download_url(url, path)
        basename = os.path.basename(url)
        archive = os.path.join(path, basename)
        print(" > Extracting archive file...")
        extract_archive(archive)


def load_file_metas(root_path:str, wavs_path="wav48_silence_trimmed", mic="mic1", ignored_speakers=None):
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
            items.append({"text": text, "audio": wav_file, "speaker": "VCTK_" + speaker_id, "root_path": root_path})
        else:
            print(f" [!] wav files don't exist - {wav_file}")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download and resample VCTK dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--current_path", type=str, default=None, required=True, help="Path of the folder containing the audio files to resample", )
    parser.add_argument("--sample_rate", type=int, default=16000, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--resample_threads", type=int, default=10, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    VCTK_DOWNLOAD_PATH = os.path.join(args.current_path, "VCTK")

    print(">>> Downloading VCTK dataset:")
    download_vctk(VCTK_DOWNLOAD_PATH)
    print(">>> resampling VCTK dataset:")
    resample_files(VCTK_DOWNLOAD_PATH, args.sample_rate, file_ext="flac", n_jobs=args.resample_threads)

