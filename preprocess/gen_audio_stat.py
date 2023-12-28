import argparse
import os
import pickle
import sys
import penn
from tqdm import tqdm
from speaker.speaker_encoder import SpeakerEncoder
import soundfile as sf

import torch
from config.config import VitsConfig
from dataset.basic_dataset import get_metas_from_filelist
from util.mel_processing import load_audio, wav_to_mel, wav_to_spec, spec_to_mel

###
# generate audio pitch, duration and save to file
###

# def get_text(text, config):
#     text_norm = text_to_sequence(text, config.data.text_cleaners)
#     if config.data.add_blank:
#         text_norm = commons.intersperse(text_norm, 0)
#     text_norm = torch.LongTensor(text_norm)
#     return text_norm

# def gen_vits_model():
#     current_path = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(os.path.join(current_path, "reference/vits"))  # change the path to vits path
#     config = utils.get_hparams_from_file("./configs/vctk_base.json")
#
#     vits_model = SynthesizerTrn(
#         len(symbols),
#         config.data.filter_length // 2 + 1,
#         config.train.segment_size // config.data.hop_length,
#         n_speakers=config.data.n_speakers,
#         **config.model
#     ).cuda()
#     # _ = vits_model.train()
#     _ = vits_model.eval()
#
#     print("loading pretrained checkpoint")
#     _ = utils.load_checkpoint("../vits_pretrained_vctk.pth", vits_model, None)
#
#     os.chdir(current_path)  # change the path back
#     return vits_model, config

#TODO: maybe it's broken already. the sampling_rate of vits is different from naturaltts
# def gen_duration_using_vits(vits_model:torch.nn.Module, config, text:str, spec:torch.Tensor, sid=4):
#     current_path = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(os.path.join(current_path, "reference/vits"))  # change the path to vits path
#
#     x_text = get_text(text, config)
#     with torch.no_grad():
#         x_text = x_text.cuda().unsqueeze(0)
#         x_text_lengths = torch.LongTensor([x_text.size(1)]).cuda()
#         y_spec = spec.cuda()
#         y_spec_lengths = torch.LongTensor([y_spec.size(2)]).cuda()
#         sid = torch.LongTensor([sid]).cuda()
#
#         audio1, _, attn1, _, _, _, _ = vits_model.forward(
#             x=x_text,
#             x_lengths=x_text_lengths,
#             y=y_spec,
#             y_lengths=y_spec_lengths,
#             sid=sid
#         )
#         audio1 = audio1[0, 0].data.cpu().float().numpy()
#         sf.write("D:/project/canoSpeech/output/test1.wav", audio1, 22050)
#
#         audio2, attn2, y_mask, _ = vits_model.infer(x_text, x_text_lengths, sid=sid, noise_scale=1, noise_scale_w=1, length_scale=1)
#         audio2 = audio2[0,0].data.cpu().float().numpy()
#         sf.write("D:/project/canoSpeech/output/test2.wav", audio2, 22050)
#
#     duration = attn1.sum(2)
#     os.chdir(current_path)  #change the path back
#     return audio2, duration


def main(args):
    config = VitsConfig()
    config.load_json(args.config)
    dataset_config = config.dataset_config.datasets[0]

    # vits, vits_config = gen_vits_model()

    speaker_encoder = SpeakerEncoder(
        config_path=args.speaker_config,
        model_path=args.speaker_model,
        use_cuda=torch.cuda.is_available(),
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
        wav, sr = load_audio(path)

        # spec = wav_to_spec(wav.unsqueeze(0), config.audio.fft_size, config.audio.hop_length, config.audio.win_length)

        # _, duration = gen_duration_using_vits(vits, vits_config, text, spec)

        # Infer pitch and periodicity
        # pitch, periodicity = penn.from_audio(
        #     audio=wav,
        #     sample_rate=config.audio.sample_rate,
        #     hopsize=config.audio.hop_length / config.audio.sample_rate,
        #     fmin=config.audio.pitch_fmin,
        #     fmax=config.audio.pitch_fmax,
        #     checkpoint=args.pitch_checkpoint,
        #     batch_size=256,
        #     interp_unvoiced_at=0.065,
        #     gpu=0
        # )
        # pitch = pitch.cpu().squeeze().numpy()
        # # periodicity = periodicity.cpu().squeeze().numpy()
        #
        # # align pitch and spectrogram length
        # refined_pitch = torch.ones([spec.shape[2]]) * pitch[0]
        # if spec.shape[2] > pitch.shape[0]:
        #     print(f"pitch is {spec.shape[2]-pitch.shape[0]} shorter than spectrogram")
        #     left = int((spec.shape[2] - pitch.shape[0]) / 2)
        #     right = pitch.shape[0] + left
        #     refined_pitch[left:right] = torch.FloatTensor(pitch)
        #
        # elif spec.shape[2] < pitch.shape[0]:
        #     print(f"pitch is {pitch.shape[0]-spec.shape[2]} longer than spectrogram")
        #     left = int((pitch.shape[0] - spec.shape[2]) / 2)
        #     right = spec.shape[2] + left
        #     refined_pitch = torch.FloatTensor(pitch[left:right])
        #
        # else:
        #     refined_pitch = pitch

        # speaker embedding
        speaker_embed = speaker_encoder.compute_embedding_from_waveform(wav)
        speaker_embed = speaker_embed.squeeze(0)
        speaker_embed = speaker_embed.cpu().float().numpy()

        obj = {
            "text": text,
            # "pitch": refined_pitch,
            # "duration": duration,
            "speaker": speaker_embed
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
    parser.add_argument("--config", type=str, default="../config/vits_aishell.json")
    parser.add_argument("--speaker_model", type=str, default="../speaker/speaker_encoder_model.pth.tar")
    parser.add_argument("--speaker_config", type=str, default="../speaker/speaker_encoder_config.json")
    parser.add_argument("--pitch_checkpoint", type=str, default="D:/dataset/VCTK/fcnf0++.pt")
    parser.add_argument("--refresh", type=bool, default=False)
    args = parser.parse_args()

    args.refresh = True
    main(args)


    # # linux config
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="/home/cano/canoSpeech/config/vits_vctk.json")
    # parser.add_argument("--speaker_model", type=str, default="/home/cano/dataset/VCTK/model_se.pth.tar")
    # parser.add_argument("--speaker_config", type=str, default="/home/cano/dataset/VCTK/config_se.json")
    # parser.add_argument("--pitch_checkpoint", type=str, default="/home/cano/dataset/VCTK/fcnf0++.pt")
    # parser.add_argument("--refresh", type=bool, default=False)
    # args = parser.parse_args()
    #
    # args.refresh = True
    # main(args)
