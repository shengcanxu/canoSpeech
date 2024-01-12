import argparse
import datetime
import os
import platform
import numpy as np
import whisper
import json

def process_snr(whisper_model, audio_folder:str, text_folder:str):
    for file in os.listdir(audio_folder):
        if not file.endswith('.mp3'): continue

        audio_path = os.path.join(audio_folder, file)
        text_path = os.path.join(text_folder, file.replace(".mp3", ".json"))
        if os.path.exists(text_path): 
            print(f"skip {file}")
            continue

        print(f'{datetime.datetime.now().strftime("%H:%M:%S")}: start {file}')
        # snr_result = whisper_model.transcribe(audio_path)
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")}: finished {file}')

        snr_result2 = decode_audio(whisper_model, audio_path)
        print(snr_result2)

        # jsonstr = json.dumps(snr_result)
        # with open(text_path, "w") as fp:
        #     fp.write(jsonstr)
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")}: finished {file}')

def decode_audio(whisper_model, audio_path:str):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    # detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    return result


if __name__ == "__main__":
    if platform.system() == "Windows":
        parser = argparse.ArgumentParser()
        parser.add_argument("--audio_folder", type=str, default="D:/dataset/librivox/downloads/")
        parser.add_argument("--text_folder", type=str, default="D:/dataset/librivox/texts/")
        parser.add_argument("--download_root", type=str, default="D:/models/whisper/")
        parser.add_argument("--model_name", type=str, default="tiny")
        args = parser.parse_args()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--audio_folder", type=str, default="/home/cano/dataset/librivox/downloads/")
        parser.add_argument("--text_folder", type=str, default="/home/cano/dataset/librivox/texts/")
        parser.add_argument("--download_root", type=str, default="/home/cano/dataset/models/whisper/")
        parser.add_argument("--model_name", type=str, default="large-v3")
        args = parser.parse_args()
    
    print("loading model...")
    model = whisper.load_model(args.model_name, download_root=args.download_root)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    process_snr(model, args.audio_folder, args.text_folder)
