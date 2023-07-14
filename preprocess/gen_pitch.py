import argparse

from config.config import VitsConfig
from dataset.dataset import get_metas_from_filelist
from util.audio_processor import AudioProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/vits.json")
    args = parser.parse_args()

    config = VitsConfig()
    config.load_json(args.config)
    dataset_config = config.dataset_config

    processor = AudioProcessor(
        hop_length=config.audio.hop_length,
        win_length=config.audio.win_length,
        sample_rate=config.audio.sample_rate,
        mel_fmin=config.audio.mel_fmin,
        mel_fmax=config.audio.mel_fmax,
        fft_size=config.audio.fft_length,
        num_mels=config.audio.num_mels,
        pitch_fmax=config.audio.pitch_fmax,
        pitch_fmin=config.audio.pitch_fmin,
        verbose=True
    )

    train_samples = get_metas_from_filelist(dataset_config.meta_file_train)
    test_samples = get_metas_from_filelist(dataset_config.meta_file_val)
    samples = train_samples
    samples.extend(test_samples)

    for sample in samples:
        path = sample["audio"]
        print("working on {}".format(path))
        wav = processor.load_wav(path)
        pitch = processor.compute_f0(wav)
        print(pitch)
        break