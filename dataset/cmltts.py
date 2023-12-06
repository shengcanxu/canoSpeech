from glob import glob
import argparse
import os
from multiprocessing import Pool
from shutil import copytree
from tqdm import tqdm
from dataset.dataset_util import download_kaggle_dataset, extract_archive, resample_file
import pandas as pd

def load_cmlpt_metas(root_path:str, meta_file="train.csv", ignored_speakers=None):
    filepath = os.path.join(root_path, meta_file)
    # ensure there are 4 columns for every line
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()

    num_cols = len(lines[0].split("|"))  # take the first row as reference
    for idx, line in enumerate(lines[1:]):
        if len(line.split("|")) != num_cols:
            print(f" > Missing column in line {idx + 1} -> {line.strip()}")

    # load metadata
    metadata = pd.read_csv(os.path.join(root_path, meta_file), sep="|")
    assert all(x in metadata.columns for x in ["wav_filename", "transcript"])

    client_id = None if "client_id" in metadata.columns else "cmlpt_0"
    emotion_name = None if "emotion_name" in metadata.columns else "neutral"
    items = []
    not_found_counter = 0
    for row in metadata.itertuples():
        if client_id is None and ignored_speakers is not None and row.client_id in ignored_speakers:
            continue
        wav_file = os.path.join(root_path, row.wav_filename)
        if not os.path.exists(wav_file):
            not_found_counter += 1
            continue

        items.append(
            {
                "text": row.transcript,
                "audio": wav_file,
                "speaker": client_id if client_id is not None else "cmlpt_" + str(row.client_id),
                "emotion": emotion_name if emotion_name is not None else row.emotion_name,
                "root_path": root_path,
                "language": "pt"
            }
        )
    if not_found_counter > 0:
        print(f" | > [!] {not_found_counter} files not found")
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
    parser = argparse.ArgumentParser(description="download and resample CMLTTS dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--sample_rate", type=int, default=22050, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--resample_threads", type=int, default=10, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    CMLTTS_DOWNLOAD_PATH = "D:\\dataset\\CMLTTS\\train\\audio"

    print(">>> resampling CMLTTS dataset:")
    resample_files(CMLTTS_DOWNLOAD_PATH, args.sample_rate, file_ext="wav", n_jobs=args.resample_threads)

    # CMLTTS_DOWNLOAD_PATH = "D:\\dataset\\CMLTTS\\"
    # items = load_cmlpt_metas(CMLTTS_DOWNLOAD_PATH)
    # print(len(items))
    # print(items[0])
