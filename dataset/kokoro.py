import argparse
import os
from dataset.dataset_util import download_url, extract_archive

def load_kokoro_metas(root_path:str, wavs_path="wavs"):
    """
    load LJSpeech dataset https://github.com/kaiidams/Kokoro-Speech-Dataset/tree/main
    """
    meta_file = "metadata.csv"
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "kokoro"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio": wav_file, "speaker": speaker_name, "root_path": root_path, "language":"en"})
    return items

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download and resample VCTK dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--current_path", type=str, default=None, required=False, help="Path of the folder containing the audio files to resample", )
    parser.add_argument("--sample_rate", type=int, default=22050, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--resample_threads", type=int, default=10, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    # LJSpeech_DOWNLOAD_PATH = os.path.join(args.current_path, "LJSpeech")
    LJSpeech_DOWNLOAD_PATH = "D:\\dataset\\LJSpeech"

    print(">>> Downloading VCTK dataset:")
    download_ljspeech(LJSpeech_DOWNLOAD_PATH)
    print(">>> resampling VCTK dataset:")
    resample_files(LJSpeech_DOWNLOAD_PATH, args.sample_rate, file_ext="flac", n_jobs=args.resample_threads)