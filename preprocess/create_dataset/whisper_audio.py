import argparse
import datetime
import json
import os
import platform
import torch
import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def process_snr(pipe, audio_folder:str, text_folder:str):
    for file in os.listdir(audio_folder):
        if not file.endswith('.mp3'): continue

        audio_path = os.path.join(audio_folder, file)
        text_path = os.path.join(text_folder, file.replace(".mp3", ".json"))
        if os.path.exists(text_path): continue

        print(f'start {file} at {datetime.datetime.now().strftime("%H:%M:%S")}, file size: {os.path.getsize(audio_path)}')
        snr_result = pipe(audio_path, return_timestamps=True)
        print(f'end {file} at {datetime.datetime.now().strftime("%H:%M:%S")}')

        jsonstr = json.dumps(snr_result)
        with open(text_path, "w") as fp:
            fp.write(jsonstr)


if __name__ == "__main__":
    if platform.system() == "Windows":
        parser = argparse.ArgumentParser()
        parser.add_argument("--audio_folder", type=str, default="D:/dataset/librivox/test/")
        parser.add_argument("--text_folder", type=str, default="D:/dataset/librivox/texts/")
        args = parser.parse_args()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--audio_folder", type=str, default="/home/cano/dataset/librivox/downloads/")
        parser.add_argument("--text_folder", type=str, default="/home/cano/dataset/librivox/texts/")
        args = parser.parse_args()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "/home/cano/models/whisper/whisper-large-v3"

    print(f'load start at {datetime.datetime.now().strftime("%H:%M:%S")}')
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2"
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
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

    process_snr(pipe, args.audio_folder, args.text_folder)
