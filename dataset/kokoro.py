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
            text = cols[1].replace(" ", "")
            items.append({"text": text, "audio": wav_file, "speaker": speaker_name, "root_path": root_path, "language":"en"})
    return items

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download and resample kokoro dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--sample_rate", type=int, default=22050, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--resample_threads", type=int, default=10, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    # LJSpeech_DOWNLOAD_PATH = os.path.join(args.current_path, "LJSpeech")
    LJSpeech_DOWNLOAD_PATH = "D:\\dataset\\kokoro"

    # wav file is download and resample using kokoro's script. no need to do it again
