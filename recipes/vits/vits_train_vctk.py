import argparse
import os
import pickle
import platform
import random
import time
from typing import Dict, Tuple

import soundfile as sf
import torch
from config.config import VitsConfig
from coqpit import Coqpit
from dataset.basic_dataset import get_metas_from_filelist
from dataset.sampler import DistributedBucketSampler
from recipes.vits.vits_train_base import VitsTrainBase
from speaker.speaker_encoder import SpeakerEncoder
from torch import nn
from trainer import Trainer, TrainerArgs
from util.mel_processing import load_audio, wav_to_spec


class VitsTrain(VitsTrainBase):
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
        text = "Who else do you want to talk to? You can go with me today to the meeting."

        speaker_name = random.choice(["VCTK_p233", "VCTK_p246", "VCTK_p263", "VCTK_p293"])
        if platform.system() == "Windows":
            path1 = "D:/dataset/VCTK/wav48_silence_trimmed/p226/p226_214_mic1.flac.wav.pkl"
            path2 = "D:/dataset/VCTK/wav48_silence_trimmed/p272/p272_378_mic1.flac.wav.pkl"
        else:
            path1 = "/home/cano/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav.pkl"
            path2 = "/home/cano/dataset/VCTK/wav48_silence_trimmed/p273/p273_004_mic1.flac.wav.pkl"
        path = random.choice([path1, path2])
        fp = open(path, "rb")
        pickleObj = pickle.load(fp)
        speaker_embed = pickleObj["speaker"]

        path = path.replace(".pkl", ".pt")
        obj = torch.load(path)
        ref_spec = obj["spec"].unsqueeze(0).cuda()

        # path = "/home/cano/dataset/LibriTTS/train-clean-100/1594/135914/1594_135914_000003_000001.wav.pt"
        # obj = torch.load(path)
        # ref_spec = obj["spec"].unsqueeze(0).cuda()

        wav = self.inference(
            text,
            speaker_name=speaker_name,
            speaker_embed=speaker_embed,
            ref_spec=ref_spec,
            language="en"
        )
        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/test_{int(time.time())}.wav", wav, 22050)

    @torch.no_grad()
    def test_voice_conversion(self, src_wav_path, output_path=None):
        src_wav, sr = load_audio(src_wav_path)
        speaker_encoder = SpeakerEncoder(
            config_path= os.path.dirname(__file__) + "/../../speaker/speaker_encoder_config.json",
            model_path= os.path.dirname(__file__) + "/../../speaker/speaker_encoder_model.pth.tar",
            use_cuda=True
        )
        source_speaker = speaker_encoder.compute_embedding_from_waveform(src_wav)
        source_speaker = source_speaker.cuda()

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
        target_speaker = target_speaker.cuda()

        y = wav_to_spec(
            src_wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        ).cuda()
        y_lengths = torch.tensor([y.size(-1)]).cuda()
        wav, _, _ = self.generator.voice_conversion(y, y_lengths, source_speaker, target_speaker)

        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/vc_{int(time.time())}.wav", wav, 22050)

    @torch.no_grad()
    def test_voice_conversion_ref_wav(self, src_wav_path, ref_wav_path=None, output_path=None):
        src_wav, sr = load_audio(src_wav_path)
        ref_wav, sr = load_audio(ref_wav_path)

        y = wav_to_spec(
            src_wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        ).cuda()
        y_lengths = torch.tensor([y.size(-1)]).cuda()
        ref_spec = wav_to_spec(
            ref_wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False
        ).cuda()

        wav, _, _ = self.generator.voice_conversion_ref_wav(y, y_lengths, ref_spec=ref_spec)

        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/vc_{int(time.time())}.wav", wav, 22050)

def main(config_path:str):
    config = VitsConfig()
    config.load_json(config_path)
    datasets = config.dataset_config.datasets

    train_samples = get_metas_from_filelist([d.meta_file_train for d in datasets])
    test_samples = get_metas_from_filelist([d.meta_file_val for d in datasets])

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
    # trainer.fit()

    # for i in range(1, 10):
    #     train_model.test_run({"output_path": "/home/cano/output"})
    #     time.sleep(1)
    # train_model.test_voice_conversion("D:\\project\\canoSpeech\\output\\test2.wav", output_path="/home/cano/output")

    # train_model.test_voice_conversion_ref_wav(
    #     src_wav_path="/home/cano/dataset/LibriTTS/train-clean-100/1594/135914/1594_135914_000003_000001.wav",
    #     ref_wav_path="/home/cano/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav",
    #     output_path="/home/cano/output"
    # )
    train_model.test_voice_conversion_ref_wav(
        src_wav_path="D:/dataset/LibriTTS/train-clean-100/1594/135914/1594_135914_000003_000001.wav",
        ref_wav_path="D:/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav",
        output_path="D:/project/canoSpeech/output"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vits vctk train", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--config_path", type=str, default="./config/vits_vctk.json", required=False)
    args = parser.parse_args()

    # main(args.config_path)

    if platform.system() == "Windows":
        main("./config/vits_vctk.json")
    else:
        main("./config/vits_vctk_linux.json")