# use demucs to separate audio and non-audio
# demucs 非常慢，可以用这个库加速： https://github.com/sakemin/demucs_batch-multigpu/tree/main

import argparse
import sys

import torch as th
import torch.cuda
from demucs.api import Separator, save_audio
from demucs.pretrained import ModelLoadingError
from models.utils.logger import FileLogger
from util.decorator import get_time

denoiser = None
def init_model():
    try:
        # document: https://github.com/adefossez/demucs/blob/main/docs/api.md
        print("initialize demucs model...")
        denoiser = Separator(
            model="htdemucs",
            repo=None,
            device= 'cuda' if torch.cuda.is_available() else 'cpu',
            shifts=0,
            split=True,
            overlap=0.25,
            progress=True,
            jobs=1,
            segment=None
        )
        return denoiser
    except ModelLoadingError as error:
        FileLogger.error(error)
        return None

@get_time
def separate_audio(from_audio:str, vocal_path:str, novocal_path:str = None):
    """ use demucs to separate audio and non-audio  """
    global denoiser
    if denoiser is None:
        denoiser = init_model()
    if denoiser is None:
        FileLogger.error("initialize demucs model failed")
        return False

    # do the separation
    origin, res = denoiser.separate_audio_file(from_audio)
    kwargs = {
        "samplerate": denoiser.samplerate,
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
    parser.add_argument("--audio_path", type=str, default="", required=False, help="需要分离的音频文件的路径")
    parser.add_argument("--vocal_path", type=str, default="",  required=False, help="输出分离后人声文件路径")
    parser.add_argument("--novocal_path", type=str, default=None,  required=False, help="输出分离后非人声文件路径")
    args = parser.parse_args()

    if sys.platform == "win32":
        args.audio_path = "D:/dataset/bilibili/translate/transformat/102805877/102805877_BV1Pw411e7Zo.wav"
        args.vocal_path = "D:/dataset/bilibili/translate/transformat/102805877/vocal.wav"
    else:
        args.audio_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3"
        args.vocal_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/vocal.wav"

    separate_audio(args.audio_path, args.vocal_path, args.novocal_path)


