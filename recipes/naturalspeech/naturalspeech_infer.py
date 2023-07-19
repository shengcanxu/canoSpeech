import argparse
from config.config import NaturalSpeechConfig
from recipes.naturalspeech.naturalspeech import NaturalSpeechTrain
from util.audio_processor import AudioProcessor
from util.helper import load_checkpoint

def main(args:dict):
    config = NaturalSpeechConfig()
    config.load_json(args.config)

    audio_processor = AudioProcessor(
        hop_length=config.audio.hop_length,
        win_length=config.audio.win_length,
        sample_rate=config.audio.sample_rate,
        mel_fmin=config.audio.mel_fmin,
        mel_fmax=config.audio.mel_fmax,
        fft_size=config.audio.fft_size,
        num_mels=config.audio.num_mels,
        pitch_fmax=config.audio.pitch_fmax,
        pitch_fmin=config.audio.pitch_fmin,
        verbose=False
    )

    model = NaturalSpeechTrain(config=config)
    model = load_checkpoint(
        path=args.path,
        model=model
    )
    wav = model.infer("here is what you want")
    audio_processor.save_wav(wav.squeeze().numpy(), path="./output/test.wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/naturalspeech_vctk.json")
    parser.add_argument("--path", type=str, default="D:/project/canoSpeech/output/CanoNaturalSpeech-July-19-2023_07+45PM-1bd469b/best_model.pth")
    args = parser.parse_args()

    main(args)