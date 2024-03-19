# use demucs to separate audio and non-audio
# demucs 非常慢，可以用这个库加速： https://github.com/sakemin/demucs_batch-multigpu/tree/main
# 也可以自己更改让一个文件也可以做batch， 推荐用这种做法。需要将 revites/apply.py这个文件替换掉demucs下面的apply.py
#
# 注意：ffmpeg 做resample的时候会改变时长。 如果想不改变时长， 需要用 librosa.resample

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import librosa
import soundfile as sf
import torch.cuda
import torch.multiprocessing as mp
import torchaudio
from demucs.api import Separator, save_audio, LoadAudioError
from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio, convert_audio_channels
from demucs.pretrained import ModelLoadingError
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
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

def convert_to_mp3(wav_path:str):
    # 注意：ffmpeg 做resample的时候会改变时长。 如果想不改变时长， 需要用 librosa.resample
    outpath = wav_path.replace("_vocal.wav", "_vocal.mp3")
    wav, sr = librosa.load(wav_path, sr=44100, mono=True, offset=0.0, duration=None)
    wav = librosa.resample(wav, orig_sr=44100, target_sr=16000, fix=True, scale=False)
    sf.write(outpath, wav, 16000, format="mp3")
    print(f"save {outpath}")

    # 清理临时文件
    os.remove(wav_path)

def convert_to_mp3_manager(root_path:str, save_threads:int=1):
    time.sleep(10)
    with mp.Pool(processes=save_threads) as pool:
        while True:
            wav_paths = glob.glob(os.path.join(root_path, f"audio/**/*_vocal.wav"), recursive=True)
            for wav_path in wav_paths:
                pool.apply_async(convert_to_mp3, args=(wav_path,))
            time.sleep(60)

def _load_audio(track: Path, samplerate=44100, audio_channels=2):
    """修改与demucs.audio._load_audio()"""
    try:
        wav, sr = torchaudio.load(str(track))
    except RuntimeError as err:
        wav = None
    else:
        wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        raise LoadAudioError(f"When trying to load {str(track)} using torch load, got error!")
    return wav

class AudioDataset(Dataset):
    def __init__(self, file_paths:list):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        wav = _load_audio(Path(file_path))
        return {'wav': wav, 'path':file_path}

def separate_audios_manager(root_path:str, file_paths:list, batch_size:int, load_threads:int, save_threads:int):
    separator = init_model()
    if separator is None:
        FileLogger.error("initialize demucs model failed")
        return False

    mp.set_start_method("spawn")
    process = mp.Process(target=convert_to_mp3_manager, args=(root_path, save_threads))
    process.start()

    dataset = AudioDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=load_threads)

    try:
        with tqdm(total=len(file_paths), desc="separate audio") as pbar:
            for idx, data in enumerate(dataloader):
                file_path = data['path'][0] if type(data['path']) == list else data['path']
                wav = data['wav'][0]

                origin, res = separate_wav(separator, wav, batch_size)
                vocals = res.pop("vocals")
                # novocals = th.zeros_like(next(iter(res.values())))
                # for v in res.values():
                #     novocals += v

                kwargs = {
                    "samplerate": 44100,
                    "bitrate": 32,  # 只是在mp3的时候有效
                    "preset": 2,
                    "clip": "rescale",
                    "as_float": False,
                    "bits_per_sample": 16,
                }
                audio_path = Path(file_path)
                vocal_path = audio_path.parent / f"{audio_path.stem}_vocal.wav"
                save_audio(vocals, vocal_path, **kwargs)

                pbar.update()
    except Exception as err:
        FileLogger.error(err)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将音频的人声和背景音乐分离")
    parser.add_argument("--batch_size", type=int, default=4, required=False, help="批量大小")
    parser.add_argument("--load_threads", type=int, default=3, required=False, help="加载线程数")
    parser.add_argument("--save_threads", type=int, default=4, required=False, help="保存文件线程数")
    parser.add_argument("--audio_path", type=str, default="", required=False, help="需要分离的音频文件的路径")
    args = parser.parse_args()

    if sys.platform == "win32":
        Wenet_PATH = "D:/dataset/WenetSpeech"
        args.audio_path = "D:/dataset/bilibili/translate/transformat/102805877/102805877_BV1Pw411e7Zo.wav"
        args.vocal_path = "D:/dataset/bilibili/translate/transformat/102805877/vocal.wav"
        # args.audio_path = "D:/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3"
        # args.vocal_path = "D:/dataset/WenetSpeech/audio/train/youtube/B00000/vocal.wav"
    else:
        Wenet_PATH = "/home/cano/dataset/WenetSpeech"
        args.audio_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3"
        args.vocal_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/vocal.wav"

    separate_audios_manager(Wenet_PATH, [args.audio_path] * 20, args.batch_size, args.load_threads, args.save_threads)


