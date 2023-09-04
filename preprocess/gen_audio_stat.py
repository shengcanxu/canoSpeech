import argparse
import os
import pickle
import sys
import time
import penn
import numpy as np
import pysptk
from tqdm import tqdm
from speaker.speaker_encoder import SpeakerEncoder

sys.path.append("D:\\project\\canoSpeech\\preprocess\\reference\\vits")
import torch
from config.config import VitsConfig
from dataset.dataset import get_metas_from_filelist
from reference.vits import commons, utils
from reference.vits.models import SynthesizerTrn
from reference.vits.text import text_to_sequence
from reference.vits.text.symbols import symbols
from util.audio_processor import AudioProcessor

###
# generate audio pitch, duration and save to file
###

def get_text(text, config):
    text_norm = text_to_sequence(text, config.data.text_cleaners)
    if config.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def gen_vits_model():
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_path, "reference/vits"))  # change the path to vits path
    config = utils.get_hparams_from_file("./configs/vctk_base.json")

    vits_model = SynthesizerTrn(
        len(symbols),
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        n_speakers=config.data.n_speakers,
        **config.model
    ).cuda()
    _ = vits_model.train()

    print("loading pretrained checkpoint")
    _ = utils.load_checkpoint("../vits_pretrained_vctk.pth", vits_model, None)

    os.chdir(current_path)  # change the path back
    return vits_model, config

#TODO: maybe it's broken already. the sampling_rate of vits is different from naturaltts
def gen_duration_using_vits(vits_model:torch.nn.Module, config, text:str, spec:torch.Tensor, sid=4):
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_path, "reference/vits"))  # change the path to vits path

    x_text = get_text(text, config)
    with torch.no_grad():
        x_text = x_text.cuda().unsqueeze(0)
        x_text_lengths = torch.LongTensor([x_text.size(1)]).cuda()
        y_spec = spec.cuda().unsqueeze(0)
        y_spec_lengths = torch.LongTensor([y_spec.size(2)]).cuda()
        sid = torch.LongTensor([sid]).cuda()

        audio, _, attn, _, _, _, _ = vits_model.forward(
            x=x_text,
            x_lengths=x_text_lengths,
            y=y_spec,
            y_lengths=y_spec_lengths,
            sid=sid
        )

    duration = attn.sum(2)
    os.chdir(current_path)  #change the path back
    return audio, duration


def main(args):
    config = VitsConfig()
    config.load_json(args.config)
    dataset_config = config.dataset_config

    processor = AudioProcessor(
        hop_length=config.audio.hop_length,
        win_length=config.audio.win_length,
        sample_rate=config.audio.sample_rate,
        mel_fmin=config.audio.mel_fmin,
        mel_fmax=config.audio.mel_fmax,
        fft_size=config.audio.fft_size,
        num_mels=config.audio.num_mels,
        pitch_fmax=config.audio.pitch_fmax,
        pitch_fmin=config.audio.pitch_fmin,
        verbose=True
    )

    vits, vits_config = gen_vits_model()

    use_cuda = torch.cuda.is_available()
    speaker_encoder = SpeakerEncoder(
        config_path=args.speaker_config,
        model_path=args.speaker_model,
        use_cuda=use_cuda,
    )

    train_samples = get_metas_from_filelist(dataset_config.meta_file_train)
    test_samples = get_metas_from_filelist(dataset_config.meta_file_val)
    samples = train_samples
    samples.extend(test_samples)

    for sample in tqdm(samples):
        path = sample["audio"]
        pklpath = path + ".pkl"
        text = sample["text"]
        # if not args.refresh and os.path.exists(pklpath):
        #     continue

        # Load audio at the correct sample rate
        wav = processor.load_wav(path, sr=config.audio.sample_rate)
        audio = torch.from_numpy(wav)
        audio = audio.unsqueeze(0)
        spec = processor.spectrogram(wav)
        spec = torch.FloatTensor(spec)

        _, duration = gen_duration_using_vits(vits, vits_config, text, spec, sid=77)

        # Infer pitch and periodicity
        pitch, periodicity = penn.from_audio(
            audio=audio,
            sample_rate=config.audio.sample_rate,
            hopsize=config.audio.hop_length / config.audio.sample_rate,
            fmin=config.audio.pitch_fmin,
            fmax=config.audio.pitch_fmax,
            checkpoint="D:\\dataset\\VCTK\\fcnf0++.pt",
            batch_size=256,
            pad=True,
            interp_unvoiced_at=0.065,
            gpu=0
        )
        pitch = pitch.cpu().squeeze().numpy()
        periodicity = periodicity.cpu().squeeze().numpy()

        # align pitch and spectrogram length
        spec = processor.spectrogram(wav)
        if spec.shape[1] > pitch.shape[0] + 1:
            print("Error! pitch is shorter than spectrogram")
        elif spec.shape[1] < pitch.shape[0]:
            print("Error! pitch is longer than spectrogram")
        elif spec.shape[1] == pitch.shape[0] + 1:
            # add the last data to the end to extend the length
            pitch = np.append(pitch, pitch[-1])
            periodicity = np.append(periodicity, periodicity[-1])

        speaker_embedd = speaker_encoder.compute_embedding_from_waveform(audio)
        speaker_embedd = speaker_embedd.squeeze(0)
        speaker_embedd = speaker_embedd.cpu().float().numpy()

        obj = {
            "text": text,
            "pitch": pitch,
            "periodicity": periodicity,
            "duration": duration,
            "speaker": speaker_embedd
        }
        with open(pklpath, "wb") as fp:
            pickle.dump(obj=obj, file=fp )

def gen_text_pitch(args):
    config = VitsConfig()
    config.load_json(args.config)
    dataset_config = config.dataset_config

    train_samples = get_metas_from_filelist(dataset_config.meta_file_train)
    test_samples = get_metas_from_filelist(dataset_config.meta_file_val)
    samples = train_samples
    samples.extend(test_samples)

    for sample in tqdm(samples):
        path = sample["audio"]
        pklpath = path + ".pkl"
        text = sample["text"]
        if not os.path.exists(pklpath): continue


        with open(pklpath, "rb") as fp:
            obj = pickle.load(fp)
            duration = obj['duration']
            pitch = obj['pitch']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/vits_vctk.json")
    parser.add_argument("--speaker_model", type=str, default="D:/dataset/VCTK/model_se.pth.tar")
    parser.add_argument("--speaker_config", type=str, default="D:/dataset/VCTK/config_se.json")
    parser.add_argument("--refresh", type=bool, default=False)
    args = parser.parse_args()

    main(args)
    # gen_text_pitch(args)

    # pklpath = 'D:\\dataset\\VCTK\\wav48_silence_trimmed\\p226\\p226_089_mic1.flac.pkl'
    # with open(pklpath, "rb") as fp:
    #     obj = pickle.load(fp)
    #     print(obj)
    #     print(len(obj["pitch"]))