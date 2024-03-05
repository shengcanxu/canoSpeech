import datetime
import sys

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

whisper_pipe = None
def init_model():
    print("initialize whisper model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_id = "/home/cano/models/whisper/whisper-large-v3"
    if sys.platform == "win32":
        model_id = "D:/models/whisper/whisper-tiny"
    else:
        model_id = "/home/cano/models/whisper"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        # attn_implementation="flash_attention_2"
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=32,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    return whisper_pipe

# do SNR using whisper from huggingface
def whisper_audio(audio_path:str):
    global whisper_pipe
    if whisper_pipe is None:
        whisper_pipe = init_model()

    snr_result = whisper_pipe(audio_path, generate_kwargs={"language": "chinese"}, return_timestamps=True)
    return snr_result