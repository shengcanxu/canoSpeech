import argparse
import os
import time
from typing import Dict, Tuple

import soundfile as sf
import torch
from config.config import VitsConfig
from coqpit import Coqpit
from dataset.basic_dataset import get_metas_from_filelist
from dataset.sampler import DistributedBucketSampler
from recipes.vits.vits_train_base import VitsTrain_Base
from text import _intersperse, text_to_tokens_cn
from torch import nn
from trainer import Trainer, TrainerArgs


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

    def inference(self, text:str, speaker_id:int=None, speaker_embed=None):
        tokens = text_to_tokens_cn(text)
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
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        output, loss_dict = self.train_step(batch, criterion, optimizer_idx)
        if optimizer_idx == 0:
            spec = batch["spec"][[0]]
            spec_len = batch["spec_lens"][[0]]
            wav = self.generator.generate_z_wav(spec, spec_len)

            wav = wav[0, 0].cpu().float().numpy()
            filename = os.path.basename(batch["filenames"][0])
            sf.write(f"{self.config.output_path}/{int(time.time())}_{filename}", wav, 22050)

        return output, loss_dict

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        output_path = assets["output_path"]
        print("doing test run...")
        text = "wo3 men2 dou1 shi4 zhong1 guo2 ren2 sp wo3 ai4 jia1 xiang1 sp"

        wav = self.inference(text)
        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/test_{int(time.time())}.wav", wav, 22050)


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
    parser = argparse.ArgumentParser(description="vits baker train", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--config_path", type=str, default="./config/vits_baker.json", required=False)
    args = parser.parse_args()

    main(args.config_path)