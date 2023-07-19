import os
from typing import Optional

from dataset.download_util import download_url, extract_archive


def download_ljspeech(save_path: str):
    """Download and extract LJSpeech dataset.
    Args:
        save_path (str): path to the directory where the dataset will be stored.
        use_kaggle (bool, optional): Downloads LJSpeech dataset from kaggle. Is generally faster. Defaults to False.
    """
    os.makedirs(save_path, exist_ok=True)
    url = "http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    download_url(url, save_path)
    basename = os.path.basename(url)
    archive = os.path.join(save_path, basename)
    print(" > Extracting archive file...")
    extract_archive(archive)

def load_ljspeech_metas(root_path:str, wavs_path="wavs"):
    """
    load LJSpeech dataset https://keithito.com/LJ-Speech-Dataset/
    """
    meta_file = "metadata.csv"
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            items.append({"text": text, "audio": wav_file, "speaker": speaker_name, "root_path": root_path})
    return items