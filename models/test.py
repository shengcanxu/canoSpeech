import torch
import torchaudio

from util.decorator import get_time


@get_time
def worka():
    t = torch.ones((2, 1000000))
    for i in range(100):
        torch.save(t, f"d:/worktest/a/test{i}.npx")

@get_time
def workb():
    t = torch.ones((2, 1000000))
    for i in range(100):
        torchaudio.save(f"d:/worktest/b/test{i}.wav", t, encoding='PCM_S', sample_rate=16000, bits_per_sample=16)

worka()
workb()