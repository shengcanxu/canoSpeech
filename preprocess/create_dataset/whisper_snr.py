import argparse
import datetime
import os
import platform
import numpy as np
import whisper
import json

from preprocess.create_dataset.batch_transcribe import batch_transcribe

def process_snr(whisper_model, audio_folder:str, text_folder:str):
    tasks = []
    for file in os.listdir(audio_folder):
        if not file.endswith('.mp3'): continue

        audio_path = os.path.join(audio_folder, file)
        text_path = os.path.join(text_folder, file.replace(".mp3", ".json"))
        if os.path.exists(text_path): continue

        tasks.append({
            "audio_path": audio_path, 
            "text_path": text_path, 
            "filesize": os.path.getsize(audio_path)
        })
    
    tasks.sort(key=lambda t: t["filesize"])

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")}: start')
    audios = [t["audio_path"] for t in tasks[0:16]]
    snr_result = batch_transcribe(whisper_model, audios)

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")}: finished')
    whisper_model.transcribe(audios[0])

    # jsonstr = json.dumps(snr_result)
    # with open(text_path, "w") as fp:
    #     fp.write(jsonstr)
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")}: finished')


if __name__ == "__main__":
    if platform.system() == "Windows":
        parser = argparse.ArgumentParser()
        parser.add_argument("--audio_folder", type=str, default="D:/dataset/librivox/test/")
        parser.add_argument("--text_folder", type=str, default="D:/dataset/librivox/texts/")
        parser.add_argument("--download_root", type=str, default="D:/models/whisper/")
        parser.add_argument("--model_name", type=str, default="tiny")
        args = parser.parse_args()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--audio_folder", type=str, default="/home/cano/dataset/librivox/downloads/")
        parser.add_argument("--text_folder", type=str, default="/home/cano/dataset/librivox/texts/")
        parser.add_argument("--download_root", type=str, default="/home/cano/models/whisper/")
        parser.add_argument("--model_name", type=str, default="large-v3")
        args = parser.parse_args()
    
    print("loading model...")
    model = whisper.load_model(args.model_name, download_root=args.download_root)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    process_snr(model, args.audio_folder, args.text_folder)
