from glob import glob
import argparse
import glob
import os
from multiprocessing import Pool
from shutil import copytree
from tqdm import tqdm
from pydub import AudioSegment

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def load_baker_metas(root_path:str, wavs_path="Wave"):
    """
    load baker dataset file meta
    """
    items = []
    meta_files_path = f"{os.path.join(root_path,'ProsodyLabeling')}/000001-010000.txt"
    wav_folder = f"{root_path}/{wavs_path}/"
    fo = open(meta_files_path, "r+", encoding="utf-8")
    while True:
        try:
            text = fo.readline().strip()
            pinyin_str = fo.readline().strip()
            if text is None or text == "" or pinyin_str is None or pinyin_str == "":
                break

            fileidx, text = text.split("\t")
            text = text.replace("#1", "").replace("#2", "").replace("#3", "").replace("#4", "")
            wav_file = wav_folder + fileidx + ".wav"

            pinyins = pinyin_str.split()
            len_pinyin = len(pinyins)
            new_pinyins = []
            idx = 0
            for word in text:
                if is_chinese(word):
                    if word == '儿' and (idx >= len_pinyin or (pinyins[idx] != 'er2' and pinyins[idx] != 'er5' )):  # 跳过儿化音
                        continue
                    if idx >= len_pinyin:
                        print(f"error! idx is larger than length of pinyins")
                        break
                    new_pinyins.append(pinyins[idx])
                    idx += 1
                else:
                    new_pinyins.append(word)

            if idx == len_pinyin:
                pinyin = ' '.join(new_pinyins)
                items.append({"text": text, "pinyin":pinyin, "audio": wav_file, "speaker": "baker", "root_path": root_path, "language":"zh"})
            else:
                print(f"error on {text}")
                #TODO: 需要处理中英混合的情况

        except Exception as e:
            print(e)
            break

    return items

# https://github.com/jaywalnut310/vits/issues/132
# resample should use ffmpeg or sox.
# if there are some blank audio at the begining or end, use librosa to trim it
def resample_file(func_args):
    filename, output_sr, file_ext = func_args
    audio = AudioSegment.from_file(filename, format=file_ext)
    audio.export(filename, format="wav", parameters=["-ar", "22050"])

def resample_files(input_dir, output_sr, output_dir=None, file_ext="wav", n_jobs=10):
    """
    change all the files to output_sr. sr = sample rate
    """
    if output_dir:
        print("Recursively copying the input folder...")
        copytree(input_dir, output_dir)
        input_dir = output_dir

    print("Resampling the audio files...")
    audio_files = glob.glob(os.path.join(input_dir, f"Wave/*.{file_ext}"), recursive=True)
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
    parser = argparse.ArgumentParser(description="download and resample Baker dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--sample_rate", type=int, default=22050, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--resample_threads", type=int, default=10, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    BAKER_DOWNLOAD_PATH = "D:/dataset/baker/"
    print(">>> resampling Baker dataset:")
    # resample_files(BAKER_DOWNLOAD_PATH, args.sample_rate, n_jobs=args.resample_threads)

    load_baker_metas(BAKER_DOWNLOAD_PATH)