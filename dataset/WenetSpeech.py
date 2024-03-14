import argparse
import json
import os
import glob
from multiprocessing import Pool
from pydub import AudioSegment
from tqdm import tqdm
import soundfile as sf

from models.speaker_diarization import speaker_diarization


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
        audio_path = os.path.join(root_path, audio_json["path"])
        if not os.path.exists(audio_path):
            print(f" [!] audio file don't exist - {audio_path}")
            return

        audio = AudioSegment.from_file(audio_path)
        folder = audio_path.replace(".opus", "")
        os.makedirs(folder, exist_ok=True)

        # for index, segment in enumerate(audio_json["segments"]):
        #     begin_time = segment.get("begin_time", 0)
        #     end_time = segment.get("end_time", 0)
        #
        #     sub_audio = audio[begin_time * 1000:end_time * 1000]
        #     sub_audio_path = os.path.join(folder, f"{index}.mp3")
        #     sub_audio.export(sub_audio_path,  format="mp3", parameters=["-ar", "16000", "-ac", "1", "-b:a", "32k"])

def split_audios(root_path:str, sample_rate:int, n_jobs=10):
    """
    split audio files to multiple audios
    """
    json_files = glob.glob(os.path.join(root_path, f"text/**/*.json"), recursive=True)
    func_args = list(zip(json_files, [root_path] * len(json_files)))
    # with Pool(processes=n_jobs) as p:
    #     with tqdm(total=len(json_files)) as pbar:
    #         for _, _ in enumerate(p.imap_unordered(split_audio, func_args)):
    #             pbar.update()
    split_audio(func_args[0])

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
                print(f" [!] {json_path} created!")
            pbar.update()

def label_audio_with_speaker(root_path:str):
    """
    add speaker info to audio json
    """
    speaker_paths = glob.glob(os.path.join(root_path, f"json/**/*_spk.json"), recursive=True)

    with tqdm(total=len(speaker_paths)) as pbar:
        pbar.update()

        # add speaker label to audio json
        for speaker_path in speaker_paths:
            audio_path = speaker_path.replace("_spk.json", ".json")
            with open(audio_path, "r", encoding="utf-8") as fp:
                audio_json = json.load(fp)
            if "all_speakers" in audio_json:
                print(f"[!] {audio_path} is finished, skip!")
                continue

            with open(speaker_path, "r", encoding="utf-8") as fp:
                speaker_list = json.load(fp)
            if speaker_list is None or len(speaker_list) == 0:
                print(f" [!] {speaker_path} is empty, skip!")
                continue

            audio_json = _align_audio_speakers(audio_json, speaker_list)

            # save back audio json to file
            with open(audio_path, "w", encoding="utf-8") as fp:
                json.dump(audio_json, fp, indent=2, ensure_ascii=False)

def _align_audio_speakers(audio_json:dict, speaker_list:list):
    speaker_list.sort(key=lambda x: x["start"])
    speaker_pos = 0
    segments = audio_json["segments"]
    segments.sort(key=lambda x: x["begin_time"])
    for segment in segments:
        if speaker_pos > len(speaker_list): break

        # find speaker list
        speakers = []
        while speaker_list[speaker_pos]["stop"] < segment["begin_time"]:
            speaker_pos += 1
        pos = speaker_pos
        while speaker_list[pos]["start"] <= segment["end_time"]:
            if speaker_list[pos]["stop"] - speaker_list[pos]["start"] >= 0.3:  # at least 0.2s
                speakers.append(speaker_list[pos])
            pos += 1
        segment["speaker_list"] = speakers

        speaker_names = set([s["speaker"] for s in speakers])
        segment["speaker_names"] = list(speaker_names)

    # add all speakers to audio json
    all_speakers = set([s["speaker"] for s in speaker_list])
    all_speakers = list(all_speakers)
    all_speakers.sort()
    audio_json["all_speakers"] = all_speakers
    return audio_json

def gen_split_audios_json(root_path:str):
    speaker_paths = glob.glob(os.path.join(root_path, f"json/**/*_spk.json"), recursive=True)

    with tqdm(total=len(speaker_paths)) as pbar:
        pbar.update()
        for speaker_path in speaker_paths:
            audio_path = speaker_path.replace("_spk.json", ".json")
            split_path = audio_path.replace(".json", "_split.json")
            if os.path.exists(split_path):
                print(f" [!] {split_path} already exists, skip!")
                pbar.update()
                continue

            with open(audio_path, "r", encoding="utf-8") as fp:
                audio_json = json.load(fp)
                if len(audio_json["segments"]) > 0:
                    split_json = _gen_split_audio(audio_json)
                    with open(split_path, "w", encoding="utf-8") as fp:
                        json.dump(split_json, fp, indent=2, ensure_ascii=False)

def _gen_split_audio(audio_json:dict):
    """ 根据发言者和持续时间进行分组。分组的规则是：首先根据发言者对片段进行初步分组，然后将长度不超过10秒的片段组合成一个大段。 """
    groups = []
    group = []
    speaker_names = audio_json["segments"][0]["speaker_names"]
    for segment in audio_json["segments"]:
        if len(segment["speaker_names"]) == 0: continue
        if len(segment["speaker_names"]) == 1 and len(speaker_names) == 1 and speaker_names[0] == segment["speaker_names"][0]:  # same speaker
            group.append(segment)
        else:
            groups.append(group)
            group = [segment]
            speaker_names = segment["speaker_names"]

    # 分割
    splits = []
    for group in groups:
        if len(group[0]["speaker_names"]) > 1: continue  # 略去多人

        split = []
        for segment in group:
            split.append(segment)
            if segment["end_time"] - split[0]["begin_time"] >= 10:
                splits.append(split)
                split = []
        if len(split) > 0:
            splits.append(split)

    # speaker统计信息
    speakers = list(set([split[0]["speaker_names"][0] for split in splits]))
    speakers = {speaker: {'audios': 0, 'duration':0} for speaker in speakers}

    segments = []
    for split in splits:
        name = split[0]["speaker_names"][0]
        segment = {
            'text': '，'.join([s["text"] for s in split]),
            'begin_time': split[0]["begin_time"],
            'end_time': split[-1]["end_time"],
            'duration': round(split[-1]["end_time"] - split[0]["begin_time"], 2),
            'segments': split,
            'speaker': name
        }
        segments.append(segment)
        speakers[name]['audios'] += 1
        speakers[name]['duration'] += segment["duration"]
    audio_json["segments"] = segments
    audio_json["split_duration"] = round(sum([s["duration"] for s in audio_json["segments"]]), 2)
    audio_json["all_speakers"] = speakers

    return audio_json

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

    # print("change to mp3")
    # change_to_mp3s(Wenet_PATH, n_jobs=args.threads)

    # print("speaker diarization")
    # create_speaker_diarization(Wenet_PATH)

    # print("label audio with speaker")
    # label_audio_with_speaker(Wenet_PATH)

    print("generate split audio json file")
    gen_split_audios_json(Wenet_PATH)

    # print("split files... ")
    # split_audios(Wenet_PATH, args.sample_rate, n_jobs=args.threads)


# ffmpeg -i Y0000000000_--5llN02F84.opus -ar 16000 -ac 1 -b:a 32k -acodec libmp3lame test.mp3