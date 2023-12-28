from glob import glob
import argparse
import os
from multiprocessing import Pool
from shutil import copytree
from tqdm import tqdm
from dataset.dataset_util import resample_file

def load_aishell3_metas(root_path:str):
    items = []
    meta_files_path = f"{root_path}/train/content.txt"
    fr = open(meta_files_path, "r+", encoding="utf-8")

    while True:
        line = fr.readline().strip()
        if line is None or line == "":
            break

        parts = line.split('	')
        filename = parts[0]
        text_str = parts[1]

        texts = text_str.split(' ')
        texts_len = len(texts)
        if texts_len % 2 == 1:  # 文字+拼音
            continue
        words = [texts[i] for i in range(0, texts_len, 2)]
        pinyins = [texts[i] for i in range(1, texts_len, 2)]

        text = "".join(words)
        pinyin = " ".join(pinyins)
        wav_file = f"{root_path}/train/wav/{filename[0:7]}/{filename}"
        sid = filename[3:7]
        items.append({"text": text, "pinyin":pinyin, "audio": wav_file, "speaker": "aishell_"+sid, "root_path": root_path, "language":"zh"})
    return items

def resample_files(input_dir, output_sr, output_dir=None, file_ext="wav", n_jobs=8):
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
    parser = argparse.ArgumentParser(description="resample AISHELL3 dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--sample_rate", type=int, default=22050, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--resample_threads", type=int, default=8, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    # AISHELL3_DOWNLOAD_PATH = "D:/dataset/AISHELL3/train/wav"
    #
    # print(">>> resampling AISHELL3 dataset:")
    # resample_files(AISHELL3_DOWNLOAD_PATH, args.sample_rate, file_ext="wav", n_jobs=args.resample_threads)

    LibriTTS_DOWNLOAD_PATH = "D:/dataset/AISHELL3"
    items = load_aishell3_metas(LibriTTS_DOWNLOAD_PATH)
    print(len(items))
    print(items[0])