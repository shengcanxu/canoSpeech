# use demucs to separate audio and non-audio
# demucs 非常慢，可以用这个库加速： https://github.com/sakemin/demucs_batch-multigpu/tree/main
# 也可以自己更改让一个文件也可以做batch， 推荐用这种做法。需要将 revites/apply.py这个文件替换掉demucs下面的apply.py

import argparse
import subprocess
import sys
from pathlib import Path

from demucs.apply import apply_model
from torch.nn import functional as F
import torch as th
import torch.cuda
import torchaudio
from demucs.api import Separator, save_audio, LoadAudioError
from demucs.audio import AudioFile, convert_audio
from demucs.pretrained import ModelLoadingError
from models.utils.logger import FileLogger
from util.decorator import get_time

separator = None
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
            progress=True,
            jobs=1,
            segment=7.8
        )
        return separator
    except ModelLoadingError as error:
        FileLogger.error(error)
        return None

def separate_audio_file(file_path:str, batch_size:int):
    global separator
    if separator is None:
        separator = init_model()
    if separator is None:
        FileLogger.error("initialize demucs model failed")
        return False

    wav = separator._load_audio(Path(file_path))

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

    # wav = _load_audio(Path(file_path))
    # channels, wav_length = wav.shape
    #
    # samplerate = separator.samplerate
    # segment = separator.segment
    # segment_length: int = int(samplerate * segment)
    # stride = int((1 - separator.overlap) * segment_length)
    # offsets = range(0, wav_length, stride)
    # scale = float(format(stride / samplerate, ".2f"))
    #
    # out = torch.zeros(channels, wav_length)
    # batch_offsets = range(0, wav_length, stride * batch_size)
    # for offset in batch_offsets:
    #     # 组建batch
    #     seg_wav = wav[:, offset:offset + segment_length * batch_size]
    #     if offset + segment_length * batch_size > wav_length:
    #         seg_wav = F.pad(seg_wav, (0, segment_length * batch_size - seg_wav.shape[-1]), mode="constant")
    #     seg_wav = seg_wav.unsqueeze(0)
    #     split_wavs = torch.split(seg_wav, batch_size, dim=-1)
    #     seg_wav = torch.stack(split_wavs, dim=0)
    #
    #     # 处理并还原为原来的长度
    #     seg_out = separator(seg_wav)
    #     split_wavs = torch.split(seg_out, batch_size, dim=0)
    #     seg_out = torch.cat(split_wavs, dim=-1)
    #     seg_out = seg_out.squeeze(0)
    #
    #     out[:, offset:offset + segment_length * batch_size] += seg_out
@get_time
def separate_audio(from_audio:str, batch_size:int, vocal_path:str, novocal_path:str = None):
    """ use demucs to separate audio and non-audio  """

    origin, res = separate_audio_file(from_audio, batch_size)
    kwargs = {
        "samplerate": separator.samplerate,
        "bitrate": 320,
        "preset": 2,
        "clip": "rescale",
        "as_float": False,
        "bits_per_sample": 16,
    }
    vocals = res.pop("vocals")
    save_audio(vocals, vocal_path, **kwargs)

    if novocal_path is not None:
        novocals = th.zeros_like(next(iter(res.values())))
        for v in res.values():
            novocals += v
        save_audio(novocals, novocal_path, **kwargs)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将音频的人声和背景音乐分离")
    parser.add_argument("--batch_size", type=int, default=4, required=False, help="批量大小")
    parser.add_argument("--audio_path", type=str, default="", required=False, help="需要分离的音频文件的路径")
    parser.add_argument("--vocal_path", type=str, default="",  required=False, help="输出分离后人声文件路径")
    parser.add_argument("--novocal_path", type=str, default=None,  required=False, help="输出分离后非人声文件路径")
    args = parser.parse_args()

    if sys.platform == "win32":
        args.audio_path = "D:/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3"
        args.vocal_path = "D:/dataset/WenetSpeech/audio/train/youtube/B00000/vocal.wav"
    else:
        args.audio_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3"
        args.vocal_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/vocal.wav"

    separate_audio(args.audio_path, args.batch_size, args.vocal_path, args.novocal_path)


