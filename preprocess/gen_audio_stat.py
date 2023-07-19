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

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def gen_vits_model():
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_path, "reference/vits"))  # change the path to vits path
    hps = utils.get_hparams_from_file("./configs/vctk_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()

    print("loading pretrained checkpoint")
    _ = utils.load_checkpoint("../vits_pretrained_vctk.pth", net_g, None)

    os.chdir(current_path)  # change the path back
    return net_g, hps

def gen_duration_using_vits(net_g:torch.nn.Module, hps, text:str, sid=4):
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(current_path, "reference/vits"))  # change the path to vits path

    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([sid]).cuda()
        audio, attn, _, _ = net_g.infer(
            x_tst,
            x_tst_lengths,
            sid=sid,
            noise_scale=.667,
            noise_scale_w=0.8,
            length_scale=1
        )

        audio = audio[0, 0].data.cpu().float().numpy()
        duration = attn.sum(2).flatten()
        duration = duration.cpu().int().numpy()

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
        if not args.refresh and os.path.exists(pklpath):
            continue

        # Load audio at the correct sample rate
        audio = penn.load.audio(path)
        hopsize = config.audio.hop_length / config.audio.sample_rate
        # Infer pitch and periodicity
        pitch, periodicity = penn.from_audio(
            audio=audio,
            sample_rate=config.audio.sample_rate,
            hopsize=hopsize,
            fmin=config.audio.pitch_fmin,
            fmax=config.audio.pitch_fmax,
            checkpoint="D:\\dataset\\VCTK\\fcnf0++.pt",
            batch_size=128,
            pad=True,
            interp_unvoiced_at=0.065,
            gpu=0)
        pitch = pitch.cpu().squeeze().numpy()

        _, duration = gen_duration_using_vits(vits, vits_config, text, sid=77)
        # processor.save_wav(audio, path="../output/test2.wav", sr=22050)

        speaker_embedd = speaker_encoder.compute_embedding_from_waveform(audio)
        speaker_embedd = speaker_embedd.squeeze(0)
        speaker_embedd = speaker_embedd.cpu().float().numpy()

        obj = {
            "text": text,
            "pitch": pitch,
            "duration": duration,
            "speaker": speaker_embedd
        }
        with open(pklpath, "wb") as fp:
            pickle.dump(obj=obj, file=fp )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/vits.json")
    parser.add_argument("--speaker_model", type=str, default="D:/dataset/VCTK/model_se.pth.tar")
    parser.add_argument("--speaker_config", type=str, default="D:/dataset/VCTK/config_se.json")
    parser.add_argument("--refresh", type=bool, default=False)
    args = parser.parse_args()

    main(args)