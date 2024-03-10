import argparse
import json
import os
import glob
import time
from multiprocessing import Pool
from pydub import AudioSegment
from tqdm import tqdm
import soundfile as sf

from preprocess.speaker_diarization import speaker_diarization


def load_wenet_train_metas(root_path:str):
    return _load_wenet_metas(root_path, wavs_path="train")

def load_wenet_test_metas(root_path:str):
    return _load_wenet_metas(root_path, wavs_path="test_net")

def _load_wenet_metas(root_path: str, wavs_path="train"):
    """
    load wennet_speech metas
    """
    items = []
    json_files = glob.glob(os.path.join(root_path, f"text/{wavs_path}/**/*.cmb.json"), recursive=True)
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
    json_path = audio["path"].replace(".opus", ".json").replace("audio", "json", 1)
    json_path = os.path.join(root_path,  json_path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # groups = []
    # group = []
    # total_duration = 0
    # for segment in audio["segments"]:
    #     group.append(segment)
    #     duration = segment["end_time"] - segment["begin_time"]
    #     total_duration += duration
    #     if total_duration >= 10:
    #         total_duration = 0
    #         groups.append({
    #             'text': '，'.join([s["text"] for s in group]),
    #             'begin_time': group[0]["begin_time"],
    #             'end_time': group[-1]["end_time"],
    #             'segments': group
    #         })
    #         group = []
    #
    # audio["segments"] = groups
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

def change_to_mp3(func_arg):
    audio_path, root_path = func_arg
    mp3_path = audio_path.replace(".opus", ".mp3")
    if os.path.exists(mp3_path):
        print(f" [!] {mp3_path} already exists, skip!")
        return

    data, sr = sf.read(audio_path, dtype='float32')
    sf.write(mp3_path, data, samplerate=16000, format="mp3")

def change_to_mp3s(root_path:str, n_jobs=10):
    audio_paths = glob.glob(os.path.join(root_path, f"audio/**/*.opus"), recursive=True)
    func_args = list(zip(audio_paths, [root_path] * len(audio_paths)))
    with Pool(processes=n_jobs) as p:
        with tqdm(total=len(audio_paths)) as pbar:
            for _, _ in enumerate(p.imap_unordered(change_to_mp3, func_args)):
                pbar.update()
    # change_to_mp3(func_args[0])

def create_speaker_diarization(root_path:str):
    audio_paths = glob.glob(os.path.join(root_path, f"audio/**/*.mp3"), recursive=True)
    with tqdm(total=len(audio_paths)) as pbar:
        for audio_path in audio_paths:
            json_path = audio_path.replace(".mp3", "_spk.json").replace("audio", "json", 1)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            if os.path.exists(json_path):
                print(f" [!] {json_path} already exists, skip!")
                pbar.update()
                continue

            speaker_list = speaker_diarization(audio_path)
            with open(json_path, "w", encoding="utf-8") as fp:
                fp.write(json.dumps(speaker_list, indent=2, ensure_ascii=False))
            pbar.update()

def label_audio_with_speaker(root_path:str):
    audio_jsons = glob.glob(os.path.join(root_path, f"json/**/*.json"), recursive=True)
    speaker_jsons = [j for j in audio_jsons if j.find("_spk.json") >= 0]
    audio_jsons = [j for j in audio_jsons if j.find("_spk.json") < 0]
    print(len(speaker_jsons))
    print(len(audio_jsons))

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

    # print("change to mp3")
    # change_to_mp3s(Wenet_PATH, n_jobs=args.threads)

    # print("speaker diarization")
    # create_speaker_diarization(Wenet_PATH)

    print("label audio with speaker")
    label_audio_with_speaker(Wenet_PATH)


# ffmpeg -i Y0000000000_--5llN02F84.opus -ar 16000 -ac 1 -b:a 32k -acodec libmp3lame test.mp3