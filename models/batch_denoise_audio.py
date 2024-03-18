# use demucs to separate audio and non-audio
# demucs 非常慢，可以用这个库加速： https://github.com/sakemin/demucs_batch-multigpu/tree/main
# 也可以自己更改让一个文件也可以做batch， 推荐用这种做法。需要将 revites/apply.py这个文件替换掉demucs下面的apply.py

import argparse
import queue
import subprocess
import sys
import time
from pathlib import Path
from multiprocessing import Process, Manager
from demucs.apply import apply_model
from torch.nn import functional as F
import torch as th
import torch.cuda
import torchaudio
from demucs.api import Separator, save_audio, LoadAudioError
from demucs.audio import AudioFile, convert_audio
from demucs.pretrained import ModelLoadingError
from tqdm import tqdm
from models.utils.logger import FileLogger

def init_model():
    try:
        # document: https://github.com/adefossez/demucs/blob/main/docs/api.md
        print("initialize demucs model...")
        separator = Separator(
            model="htdemucs",
            repo=None,
            device= 'cuda' if torch.cuda.is_available() else 'cpu',
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
            jobs=1,
            segment=7.8
        )
        return separator
    except ModelLoadingError as error:
        FileLogger.error(error)
        return None

def _load_audio(track: Path, samplerate=44100, audio_channels=2):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(streams=0, samplerate=samplerate, channels=audio_channels)
    except FileNotFoundError:
        errors["ffmpeg"] = "FFmpeg is not installed."
    except subprocess.CalledProcessError:
        errors["ffmpeg"] = "FFmpeg could not read the file."

    if wav is None:
        try:
            wav, sr = torchaudio.load(str(track))
        except RuntimeError as err:
            errors["torchaudio"] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        raise LoadAudioError(
            "\n".join("When trying to load using {}, got the following error: {}".format( backend, error)
                for backend, error in errors.items()
            )
        )
    return wav

def separate_wav(separator, wav:torch.Tensor, batch_size:int):
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8
    batch_size = 4
    out = apply_model(
        separator._model,
        wav[None],
        segment=separator._segment * batch_size,
        shifts=separator._shifts,
        split=separator._split,
        overlap=separator._overlap,
        device=separator._device,
        num_workers=separator._jobs,
        progress=separator._progress,
        batch_size = batch_size
    )
    if out is None:
        raise KeyboardInterrupt
    out *= ref.std() + 1e-8
    out += ref.mean()
    wav *= ref.std() + 1e-8
    wav += ref.mean()
    return (wav, dict(zip(separator._model.sources, out[0])))

def load_audios(path_queue:queue.Queue, load_queue:queue.Queue):
    while not path_queue.empty():
        if load_queue.full():
            time.sleep(1)
            continue

        audio_path = path_queue.get()
        wav = _load_audio(Path(audio_path))
        load_queue.put((audio_path, wav))

    print(f"end load audio process")

def separate_audios(load_finished, load_queue:queue.Queue, save_queue:queue.Queue, batch_size:int, files:int):
    separator = init_model()
    if separator is None:
        FileLogger.error("initialize demucs model failed")
        return False

    with tqdm(total=files, desc="separate audio") as pbar:
        while True:
            if load_queue.empty():
                if load_finished.value is False:
                    time.sleep(1)
                    continue
                else:
                    break

            audio_path, wav = load_queue.get()
            origin, res = separate_wav(separator, wav, batch_size)
            vocals = res.pop("vocals")
            novocals = th.zeros_like(next(iter(res.values())))
            for v in res.values():
                novocals += v
            save_queue.put((audio_path, vocals, novocals))
            pbar.update()

    print(f"end separate audio process")

def save_audios(separate_finished, save_queue:queue.Queue):
    while True:
        if save_queue.empty():
            if separate_finished.value is False:
                time.sleep(1)
                continue
            else:
                break

        audio_path, vocals, novocals = save_queue.get()
        kwargs = {
            "samplerate": 44100,
            "bitrate": 320,
            "preset": 2,
            "clip": "rescale",
            "as_float": False,
            "bits_per_sample": 16,
        }
        audio_path = Path(audio_path)
        vocal_path = audio_path.parent / f"{audio_path.stem}_vocals.mp3"
        save_audio(vocals, vocal_path, **kwargs)
        print(f"saved {vocal_path}")
        # novocal_path = audio_path.parent / f"{audio_path.stem}_nonvocals.mp3"
        # save_audio(novocals, novocal_path, **kwargs)
        # print(f"saved {novocal_path}")


def separate_audios_manager(file_paths:list, batch_size:int, load_threads:int, save_threads:int):
    manager = Manager()
    path_queue = manager.Queue()
    for path in file_paths:
        path_queue.put(path)
    load_queue = manager.Queue(maxsize=batch_size * 3)
    save_queue = manager.Queue()
    load_finished = manager.Value(bool, False)
    separate_finished = manager.Value(bool, False)

    load_processes = []
    for i in range(load_threads):
        load_process = Process(target=load_audios, args=(path_queue, load_queue))
        load_process.start()
        load_processes.append(load_process)
    separate_process = Process(target=separate_audios, args=(load_finished, load_queue, save_queue, batch_size, len(file_paths)))
    separate_process.start()
    save_processes = []
    for i in range(save_threads):
        save_process = Process(target=save_audios, args=(separate_finished, save_queue))
        save_process.start()
        save_processes.append(save_process)

    for p in load_processes:
        p.join()
    load_finished.value = True
    separate_process.join()
    separate_finished.value = True
    for p in save_processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将音频的人声和背景音乐分离")
    parser.add_argument("--batch_size", type=int, default=4, required=False, help="批量大小")
    parser.add_argument("--load_threads", type=int, default=2, required=False, help="加载线程数")
    parser.add_argument("--save_threads", type=int, default=6, required=False, help="保存文件线程数")
    parser.add_argument("--audio_path", type=str, default="", required=False, help="需要分离的音频文件的路径")
    args = parser.parse_args()

    if sys.platform == "win32":
        args.audio_path = "D:/dataset/bilibili/translate/transformat/102805877/102805877_BV1Pw411e7Zo.wav"
        args.vocal_path = "D:/dataset/bilibili/translate/transformat/102805877/vocal.wav"
        # args.audio_path = "D:/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3"
        # args.vocal_path = "D:/dataset/WenetSpeech/audio/train/youtube/B00000/vocal.wav"
    else:
        args.audio_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3"
        args.vocal_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/vocal.wav"

    separate_audios_manager([args.audio_path] * 20, args.batch_size, args.load_threads, args.save_threads)


