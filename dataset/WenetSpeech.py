import argparse
import json
import os
import glob
from multiprocessing import Pool
from pydub import AudioSegment
from tqdm import tqdm

def load_wenet_train_metas(root_path:str):
    return _load_wenet_metas(root_path, wavs_path="train")

def load_wenet_test_metas(root_path:str):
    return _load_wenet_metas(root_path, wavs_path="test_net")

def _load_wenet_metas(root_path: str, wavs_path="train"):
    """
    load wennet_speech metas
    """
    items = []
    json_files = glob.glob(os.path.join(root_path, f"text/{wavs_path}/**/*.json"), recursive=True)
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as fp:
            audio_json = json.load(fp)
            for index, segment in enumerate(audio_json["segments"]):
                text = segment["text"]
                wav_file = audio_json["path"].replace(".opus", f"/{index}.mp3")
                wav_file = os.path.join(root_path, wav_file)
                speaker_id = audio_json["aid"]

                if os.path.exists(wav_file):
                    items.append({"text": text, "audio": wav_file, "speaker": speaker_id, "root_path": root_path, "language":"zh"})
                else:
                    print(f" [!] wav files don't exist - {wav_file}")
    return items

def create_json_file(audio_json:dict):
    audio, root_path = audio_json
    json_path = audio["path"].replace(".opus", ".json").replace("audio/", "text/")
    json_path = os.path.join(root_path,  json_path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    groups = []
    group = []
    total_duration = 0
    for segment in audio["segments"]:
        group.append(segment)
        duration = segment["end_time"] - segment["begin_time"]
        total_duration += duration
        if total_duration >= 10:
            total_duration = 0
            groups.append({
                'text': '，'.join([s["text"] for s in group]),
                'begin_time': group[0]["begin_time"],
                'end_time': group[-1]["end_time"],
                'segments': group
            })
            group = []

    audio["segments"] = groups
    with open(json_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(audio, indent=2, ensure_ascii=False))

def create_json_files(root_path:str, n_jobs=10):
    """
    create json files for wenet dataset
    """
    with open(os.path.join(root_path, "WenetSpeech.json"), 'r', encoding="utf-8") as fp:
        db_json = json.load(fp)

    audios = db_json["audios"]
    audio_num = len(audios)
    print(f"there are {audio_num} audios...")

    audio_jsons = list(zip(audios, [root_path] * audio_num))
    with Pool(processes=n_jobs) as p:
        with tqdm(total=audio_num) as pbar:
            for _, _ in enumerate(p.imap(create_json_file, audio_jsons)):
                pbar.update()
    # create_json_file(audio_jsons[0])

    print("finish processing wenet speech json files")

def split_audio(func_arg):
    json_path, root_path = func_arg
    with open(json_path, "r", encoding="utf-8") as fp:
        audio_json = json.load(fp)
        if "path" not in audio_json or "segments" not in audio_json:
            raise ValueError("Invalid JSON structure")

        audio_path = os.path.join(root_path, audio_json["path"])
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")

        audio = AudioSegment.from_file(audio_path)
        folder = audio_path.replace(".opus", "")
        if not os.path.exists(folder):
            os.makedirs(folder)

        for index, segment in enumerate(audio_json["segments"]):
            begin_time = segment.get("begin_time", 0)
            end_time = segment.get("end_time", 0)
            if begin_time < 0 or end_time < 0 or end_time <= begin_time:
                continue  # 跳过无效的时间段

            sub_audio = audio[begin_time * 1000:end_time * 1000]
            sub_audio_path = os.path.join(folder, f"{index}.mp3")
            sub_audio.export(sub_audio_path,  format="mp3", parameters=["-ar", "16000", "-ac", "1", "-b:a", "32k"])

def split_audios(root_path:str, sample_rate:int, n_jobs=10):
    """
    split audio files to multiple audios
    """
    json_files = glob.glob(os.path.join(root_path, f"text/**/*.json"), recursive=True)
    func_args = list(zip(json_files, [root_path] * len(json_files)))
    with Pool(processes=n_jobs) as p:
        with tqdm(total=len(json_files)) as pbar:
            for _, _ in enumerate(p.imap_unordered(split_audio, func_args)):
                pbar.update()
    # split_audio(func_args[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create json and split mp3 for WenetSpeech dataset", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--db_path", type=str, default=None, required=False, help="Path of the folder containing the audio files to resample", )
    parser.add_argument("--sample_rate", type=int, default=16000, required=False, help="the sample rate that resample the audio to")
    parser.add_argument("--threads", type=int, default=8, required=False, help="Define the number of threads used during the audio resampling")
    args = parser.parse_args()

    Wenet_PATH = "D:\\dataset\\WenetSpeech"
    # Wenet_PATH = "/home/cano/dataset/WenetSpeech"

    # print("create audio json file...")
    # create_json_files(Wenet_PATH, n_jobs=args.threads)

    # print("split files... ")
    # split_audios(Wenet_PATH, args.sample_rate, n_jobs=args.threads)

    items = load_wenet_train_metas(Wenet_PATH)
    print(items)


# ffmpeg -i Y0000000000_--5llN02F84.opus -ar 16000 -ac 1 -b:a 32k -acodec libmp3lame test.mp3