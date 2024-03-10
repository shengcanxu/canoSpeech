# speaker diarization using pyannote.audio: https://github.com/pyannote/pyannote-audio
# 由于pyannote对模型做了封装，很难缓存模型到指定cache，只能使用默认的huggingface cache
# install:
# pip install pyannote.audio
# pip install speechbrain
# 另外跟pytorch2.2.1冲突，需要用pytorch2.0.1， 安装命令如下：
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
#
# pyannote.audio的逻辑是每次都是请求一下huggingface获得配置，这样让模型非常慢， 可以修改yaml配置来将所有的模型都指向本地文件
# 1. Pipeline.from_pretrained()里不是传huggingface的id， 而是传入config.yaml里面具体的地址（这个config.yaml可以放到代码目录以便修改）
# 例如： D:/models/pyannote/models--pyannote--speaker-diarization-3.1/snapshots/0c6d72ac70c2dca2b11b236f5ca3d54d0c133109/config.yaml
# 2. 将config.yaml的segmentation 改成指向本地的segmentation下面的pytorch_model.bin文件，
# 例如：C:/Users/CanoLaptop/.cache/torch/pyannote/models--pyannote--segmentation-3.0/snapshots/f47575fb2e9be1f0b93981209ea5cba0512b3acb/pytorch_model.bin
# 3. 将config.yaml的embedding改成指向本地的embedding下面的pytorch_model.bin文件
# 例如：C:/Users/CanoLaptop/.cache/torch/pyannote/models--pyannote--wespeaker-voxceleb-resnet34-LM/snapshots/76bad8b0aadd8f951530dd6e543c59d1dcd7c62c/pytorch_model.bin

import json
import sys
import time

import torchaudio
from pyannote.audio import Pipeline
import torch
import os

diarization_pipe = None

def init_model():
    print("initialize diarization model...")
    # os.environ["http_proxy"] = "http://192.168.0.118:10809"
    # os.environ["https_proxy"] = "http://192.168.0.118:10809"

    if sys.platform == "win32":
        cache_dir = "D:/models/pyannote"
        config_path = os.path.join(os.path.dirname(__file__), "config/speaker_diarization_config.yaml")
    else:
        cache_dir = "/home/cano/models/pyannote"
        config_path = os.path.join(os.path.dirname(__file__), "config/speaker_diarization_config_linux.yaml")

    diarization_pipe = Pipeline.from_pretrained(
        config_path,
        # "pyannote/speaker-diarization-3.1",
        use_auth_token="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",
        cache_dir=cache_dir
    )
    diarization_pipe.to(torch.device("cuda"))
    print(f"model loaded at {time.time()}")
    return diarization_pipe

def speaker_diarization(audio_path:str):
    global diarization_pipe
    if diarization_pipe is None:
        diarization_pipe = init_model()

    waveform, sample_rate = torchaudio.load(audio_path)
    diarization = diarization_pipe({"waveform": waveform, "sample_rate": sample_rate})
    # diarization = diarization_pipe(audio_path)

    speaker_list = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_list.append({
            "start": round(turn.start, 2),
            "stop": round(turn.end, 2),
            "speaker": speaker
        })
    return speaker_list

if __name__ == "__main__":
    print(time.time())
    # speaker_list = speaker_diarization("D:/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3")
    # json_path = "D:/dataset/WenetSpeech/audio/train/youtube/B00000/speaker.json"
    speaker_list = speaker_diarization("/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.mp3")
    json_path = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/speaker.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(speaker_list, indent=2, ensure_ascii=False))
    print(time.time())