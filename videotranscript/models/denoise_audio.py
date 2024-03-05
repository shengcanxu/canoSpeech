import torch as th
import torch.cuda
from demucs.api import Separator, save_audio
from demucs.pretrained import ModelLoadingError
from models.utils.logger import FileLogger

separator = None
def separate_audio(from_audio:str, vocal_path:str, novocal_path:str):
    """ use demucs to separate audio and non-audio  """
    global separator
    if separator is None:
        try:
            # document: https://github.com/adefossez/demucs/blob/main/docs/api.md
            print("initialize demucs model...")
            separator = Separator(model="mdx_extra",
                                  repo=None,
                                  device= 'cuda' if torch.cuda.is_available() else 'cpu',
                                  shifts=1,
                                  split=True,
                                  overlap=0.25,
                                  progress=True,
                                  jobs=0,
                                  segment=None)
        except ModelLoadingError as error:
            FileLogger.error(error)
            return False

    # do the separation
    origin, res = separator.separate_audio_file(from_audio)
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

    novocals = th.zeros_like(next(iter(res.values())))
    for v in res.values():
        novocals += v
    save_audio(novocals, novocal_path, **kwargs)
    return True

if __name__ == "__main__":
    import sys
    print(sys.path)

    file = "D:/dataset/bilibili/translate/transformat/102805877/102805877_BV1Pw411e7Zo.wav"
