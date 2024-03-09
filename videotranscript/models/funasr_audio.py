# funasr的很多例子程序见： https://github.com/alibaba-damo-academy/FunASR/tree/main/examples
# 也可以用funasr来做speaker_diarization， 例子可以见：https://github.com/alibaba-damo-academy/3D-Speaker/tree/3dspeaker/egs/3dspeaker/speaker-diarization
# 总体来说， speaker_diarization的基本步骤是 1）将音频分成很多个segmentation ， 2）对每个segmentation做speaker_diarization，最好是用batch的方式来做

import time
from funasr import AutoModel
import sys

funasr_model = None
def init_model():
    print("initialize funasr model...")
    # model = AutoModel(
    #     model="paraformer-zh",
    #     model_revision="v2.0.4",
    #     vad_model="fsmn-vad",
    #     vad_model_revision="v2.0.4",
    #     punc_model="ct-punc-c",
    #     punc_model_revision="v2.0.4",
    #     spk_model="cam++",
    #     spk_model_revision="v2.0.2",
    # )

    if sys.platform == "win32":
        model_id = "D:/models/funasr/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        vad_model_id = "D:/models/funasr/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        punc_model_id = "D:/models/funasr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        camplus_model_id = "D:/models/funasr/speech_campplus_sv_zh-cn_16k-common"
    else:
        model_id = "/home/cano/models/funasr/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        vad_model_id = "/home/cano/models/funasr/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        punc_model_id = "/home/cano/models/funasr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        camplus_model_id = "/home/cano/models/funasr/speech_campplus_sv_zh-cn_16k-common"

    model = AutoModel(
        model=model_id,
        model_revision="v2.0.4",
        # vad_model=vad_model_id,
        # vad_model_revision="v2.0.4",
        punc_model=punc_model_id,
        punc_model_revision="v2.0.4",
        spk_model=camplus_model_id,
        spk_model_revision="v2.0.2",
    )
    return model

# do SNR using funasr model from modelscrope
def funasr_audio(audio_path:str):
    global funasr_model
    if funasr_model is None:
        funasr_model = init_model()

    res = funasr_model.generate(input=audio_path, batch_size_s=300, hotword='魔搭')
    return res

if __name__ == "__main__":

    # wav = "D:/dataset/bilibili/translate/denoise/102805877/102805877_BV1Pw411e7Zo_vocal.wav"

    # model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
    #               vad_model="fsmn-vad", vad_model_revision="v2.0.4",
    #               punc_model="ct-punc-c", punc_model_revision="v2.0.4",
    #               spk_model="cam++", spk_model_revision="v2.0.2",
    #               )

    # model_id = "D:/models/funasr/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    # vad_model_id = "D:/models/funasr/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    # punc_model_id = "D:/models/funasr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    # camplus_model_id = "D:/models/funasr/speech_campplus_sv_zh-cn_16k-common"
    # model = AutoModel(model=model_id, model_revision="v2.0.4",
    #               vad_model=vad_model_id, vad_model_revision="v2.0.4",
    #               punc_model=punc_model_id, punc_model_revision="v2.0.4",
    #               spk_model=camplus_model_id, spk_model_revision="v2.0.2",
    #               )
    # res = model.generate(input=wav,
    #             batch_size_s=300,
    #             hotword='魔搭')
    # print(res)

    print(time.time())
    # wav = "D:/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus"
    wav = "/home/cano/dataset/WenetSpeech/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus"
    res = funasr_audio(wav)
    print(res)
    print(time.time())