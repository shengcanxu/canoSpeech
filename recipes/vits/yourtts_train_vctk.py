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
from recipes.vits.vits_train_base import VitsTrain_Base
from text import text_to_tokens, _intersperse
from torch import nn
from trainer import Trainer, TrainerArgs


class YourttsTrain(VitsTrain_Base):
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

    def on_epoch_start(self, trainer):
        if self.model_config.use_speaker_encoder_as_loss:
            # freeze SpeakerEncoder layer
            for param in self.generator.speaker_encoder.encoder.parameters():
                param.requires_grad = False

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

        if platform.system() == "Windows":
            path1 = "D:/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav.pkl"
            path2 = "D:/dataset/VCTK/wav48_silence_trimmed/p273/p273_004_mic1.flac.wav.pkl"
        else:
            path1 = "/home/cano/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav.pkl"
            path2 = "/home/cano/dataset/VCTK/wav48_silence_trimmed/p273/p273_004_mic1.flac.wav.pkl"
        path = path1 if random.randint(1,10) >= 5 else path2
        fp = open(path, "rb")
        pickleObj = pickle.load(fp)
        speaker_embed = pickleObj["speaker"]

        wav = self.inference(text, speaker_embed=speaker_embed)
        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/test_{int(time.time())}.wav", wav, 22050)

# need to change lr to a low number(e.g.: 1e-8), and change scheduler_after_epoch to false
def find_lr(trainer:Trainer, config:Coqpit, steps=1000):
    losses = trainer.find_lr_fit(steps = steps)
    with open(f"{config.output_path}/lr.txt", "w") as fp:
        for loss in losses:
            fp.write(",".join([str(ll) for ll in loss]) + "\n")

def main(config_path:str):
    config = VitsConfig()
    config.load_json(config_path)
    data_config = config.dataset_config

    train_samples = get_metas_from_filelist(data_config.meta_file_train)
    test_samples = get_metas_from_filelist(data_config.meta_file_val)

    # init the model
    train_model = YourttsTrain(config=config)

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

    find_lr(trainer, config, 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yourtts vctk train", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--config_path", type=str, default="./config/yourtts_vctk.json", required=False)
    args = parser.parse_args()

    main(args.config_path)