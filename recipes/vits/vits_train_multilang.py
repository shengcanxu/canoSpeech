import argparse
import os
import platform
import random
import time
from typing import Dict, Tuple

import soundfile as sf
import torch
from config.config import VitsConfig
from dataset.basic_dataset import get_metas_from_filelist
from recipes.vits.vits_train_base import VitsTrain_Base
from torch import nn
from trainer import Trainer, TrainerArgs

class VitsTrain(VitsTrain_Base):
    """
    VITS and YourTTS model training model.
    """
    def __init__(self, config:VitsConfig ):
        super().__init__(config)

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        output, loss_dict = self.train_step(batch, criterion, optimizer_idx)
        if optimizer_idx == 0:
            num = batch["spec"].size(0)
            idx = random.randint(0, num-1)
            spec = batch["spec"][[idx]]
            spec_len = batch["spec_lens"][[idx]]
            speaker_id = batch["speaker_ids"][[idx]]
            wav = self.generator.generate_z_wav(spec, spec_len, speaker_id)

            wav = wav[0, 0].cpu().float().numpy()
            filename = os.path.basename(batch["filenames"][idx])
            sf.write(f"{self.config.output_path}/{int(time.time())}_{filename}", wav, 22050)

        return output, loss_dict

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        output_path = assets["output_path"]

        print("doing test run...")
        text1 = "Who else do you want to talk to? You can go with me today to the meeting."
        text2 = "我们都是中国人，我爱中国"
        text3 = "Sou estudante, mas não quero fazer lição de casa."
        text4 = "私たちは全員中国人で、故郷が大好きです。"
        lang = random.choice(["en", "zh", "pt", "ja"])
        if lang == "en":
            speaker_name = random.choice(["VCTK_p240", "VCTK_p260", "VCTK_p270", "VCTK_p311", "VCTK_p336"])
            wav = self.inference(text1, speaker_name=speaker_name, language="en")
        elif lang == "zh":
            wav = self.inference(text2, speaker_name="baker", language="zh")
        elif lang == "pt":
            speaker_name = random.choice(["cmlpt_2961", "cmlpt_5705", "cmlpt_6700", "cmlpt_10670", "cmlpt_5025"])
            wav = self.inference(text3, speaker_name=speaker_name, language="pt")
        elif lang == "ja":
            wav = self.inference(text4, speaker_name="kokoro", language="ja")
        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/test_{int(time.time())}.wav", wav, 22050)


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
    trainer.fit()

    # train_model.test_run({"output_path": "/home/cano/output"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vits vctk train", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--config_path", type=str, default="./config/vits_multilang.json", required=False)
    args = parser.parse_args()

    # main(args.config_path)

    if platform.system() == "Windows":
        main("./config/vits_multilang.json")
    else:
        main("./config/vits_multilang_linux.json")