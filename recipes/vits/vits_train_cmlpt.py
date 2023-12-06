import argparse
import os
import pickle
import random
import time
from typing import Dict, Tuple

import soundfile as sf
import torch
from config.config import VitsConfig
from coqpit import Coqpit
from dataset.basic_dataset import get_metas_from_filelist
from dataset.sampler import DistributedBucketSampler
from recipes.vits.vits_train_base import VitsTrain_Base
from speaker.speaker_encoder import SpeakerEncoder
from text import text_to_tokens, _intersperse
from torch import nn
from trainer import Trainer, TrainerArgs
from util.mel_processing import load_audio, wav_to_spec


class VitsTrain(VitsTrain_Base):
    """
    VITS and YourTTS model training model.
    """
    def __init__(self, config:VitsConfig):
        super().__init__(config)

    def get_sampler(self, config: Coqpit, dataset, num_gpus=1, rank=0):
        return DistributedBucketSampler(
            dataset=dataset,
            batch_size=config.batch_size,
            boundaries=[32,300,400,500,600,700,800,900,1000],
            num_replicas=num_gpus,
            rank=0,
            shuffle=True
        )

    @torch.no_grad()
    def inference(self, text:str, speaker_id:int=None, speaker_embed=None):
        tokens = text_to_tokens(text, cleaner_names=self.config.text.text_cleaners)
        if self.config.text.add_blank:
            tokens = _intersperse(tokens, 0)
        tokens = torch.LongTensor(tokens).unsqueeze(dim=0).cuda()
        x_lengths = torch.LongTensor([tokens.size(1)]).cuda()

        speaker_ids = torch.LongTensor([speaker_id]).cuda() if speaker_id is not None else None
        speaker_embed = torch.FloatTensor(speaker_embed).unsqueeze(0).cuda() if speaker_embed is not None else None

        self.generator.eval()
        wav, _, _, _ = self.generator.infer(
            tokens,
            x_lengths,
            speaker_ids=speaker_ids,
            speaker_embeds=speaker_embed,
            noise_scale=0.8,
            length_scale=1,
        )
        return wav

    @torch.no_grad()
    def inference_voice_conversion(self, reference_wav, ref_speaker_id=None, ref_speaker_embed=None, speaker_id=None, speaker_embed=None):
        y = wav_to_spec(
            reference_wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        ).cuda()
        y_lengths = torch.tensor([y.size(-1)]).cuda()
        source_speaker = ref_speaker_id if ref_speaker_id is not None else ref_speaker_embed
        target_speaker = speaker_id if speaker_id is not None else speaker_embed
        source_speaker = source_speaker.cuda()
        target_speaker = target_speaker.cuda()
        wav, _, _ = self.generator.voice_conversion(y, y_lengths, source_speaker, target_speaker)
        return wav

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        output, loss_dict = self.train_step(batch, criterion, optimizer_idx)
        if optimizer_idx == 0:
            spec = batch["spec"][[0]]
            spec_len = batch["spec_lens"][[0]]
            speaker_id = batch["speaker_ids"][[0]]
            speaker_embed = batch["speaker_embeds"][[0]]
            wav = self.generator.generate_z_wav(spec, spec_len, speaker_id, speaker_embed)

            wav = wav[0, 0].cpu().float().numpy()
            filename = os.path.basename(batch["filenames"][0])
            sf.write(f"{self.config.output_path}/{int(time.time())}_{filename}", wav, 22050)

        return output, loss_dict

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        output_path = assets["output_path"]
        print("doing test run...")
        text = "Com quem mais você quer conversar? Você pode ir comigo hoje à reunião."

        # speaker_id = random.randint(0, 9)
        path1 = "D:/dataset/CMLTTS/train/audio/9217/6390/9217_6390_000038.wav.pkl"
        path2 = "D:/dataset/CMLTTS/train/audio/3427/2564/3427_2564_000000-0001.wav.pkl"
        # path1 = "/home/cano/dataset/CMLTTS/train/audio/9217/6390/9217_6390_000038.wav.pkl"
        # path2 = "/home/cano/dataset/CMLTTS/train/audio/3427/2564/3427_2564_000000-0001.wav.pkl"
        path = path1 if random.randint(1,10) >= 5 else path2
        fp = open(path, "rb")
        pickleObj = pickle.load(fp)
        speaker_embed = pickleObj["speaker"]

        wav = self.inference(text, speaker_embed=speaker_embed)
        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/test_{int(time.time())}.wav", wav, 22050)

def test(model, filepath:str):
    # speaker embedding
    wav, sr = load_audio(filepath)
    speaker_encoder = SpeakerEncoder(
        config_path=os.path.dirname(__file__) + "/../../speaker/speaker_encoder_config.json",
        model_path=os.path.dirname(__file__) + "/../../speaker/speaker_encoder_model.pth.tar",
        use_cuda=True
    )
    speaker_embed = speaker_encoder.compute_embedding_from_waveform(wav)
    speaker_embed = speaker_embed.squeeze(0)
    speaker_embed = speaker_embed.cpu().float().numpy()

    text = "I am a student but I don't want to do any homework."
    wav = model.inference(text, speaker_embed=speaker_embed)
    wav = wav[0, 0].cpu().float().numpy()
    sf.write(f"{filepath}.test.wav", wav, 22050)

def test_voice_conversion(model, ref_wav_filepath:str):
    wav, sr = load_audio(ref_wav_filepath)
    speaker_encoder = SpeakerEncoder(
        config_path= os.path.dirname(__file__) + "/../../speaker/speaker_encoder_config.json",
        model_path= os.path.dirname(__file__) + "/../../speaker/speaker_encoder_model.pth.tar",
        use_cuda=True
    )
    source_speaker = speaker_encoder.compute_embedding_from_waveform(wav)

    path1 = "D:/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav.pkl"
    # path2 = "D:/dataset/VCTK/wav48_silence_trimmed/p273/p273_004_mic1.flac.wav.pkl"
    # path1 = "/home/cano/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav.pkl"
    # path2 = "/home/cano/dataset/VCTK/wav48_silence_trimmed/p273/p273_004_mic1.flac.wav.pkl"
    # path = path1 if random.randint(1, 10) >= 5 else path2
    path = path1
    fp = open(path, "rb")
    pickleObj = pickle.load(fp)
    target_speaker = pickleObj["speaker"]
    target_speaker = torch.from_numpy(target_speaker).unsqueeze(0)

    out_wav = model.inference_voice_conversion(
        reference_wav = wav, ref_speaker_embed = source_speaker, speaker_embed = target_speaker
    )
    out_wav = out_wav[0, 0].cpu().float().numpy()
    sf.write(f"{ref_wav_filepath}.out.wav", out_wav, 22050)

def main(config_path:str):
    config = VitsConfig()
    config.load_json(config_path)
    data_config = config.dataset_config

    train_samples = get_metas_from_filelist(data_config.meta_file_train)
    test_samples = get_metas_from_filelist(data_config.meta_file_val)

    # init the model
    train_model = VitsTrain(config=config)

    # init the trainer and train
    trainer = Trainer(
        TrainerArgs(continue_path=config.continue_path, restore_path=config.restore_path, skip_train_epoch=False),
        config,
        output_path=config.output_path,
        model=train_model,
        train_samples=train_samples,
        eval_samples=test_samples,
    )
    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vits cmltts pt train", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--config_path", type=str, default="./config/vits_cmlpt.json", required=False)
    args = parser.parse_args()

    main(args.config_path)