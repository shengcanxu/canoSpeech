import argparse
import os
import platform
import time
from typing import Dict, Tuple, List

import soundfile as sf
import torch
from torch import nn
from torch.cuda.amp import autocast

from config.config import VitsConfig
from coqpit import Coqpit
from dataset.basic_dataset import get_metas_from_filelist
from dataset.sampler import DistributedBucketSampler
from layers.discriminator import VitsDiscriminator
from layers.losses import VitsDiscriminatorLoss, VitsGeneratorLoss, VAEGeneratorLoss
from recipes.trainer_model import TrainerModelWithDataset
from recipes.vits.vits import VitsModel
from trainer import Trainer, TrainerArgs
from trainer import get_optimizer
from util.helper import segment
from util.mel_processing import wav_to_mel


class VitsTrain(TrainerModelWithDataset):
    """
    VITS and YourTTS model training model.
    """
    def __init__(self, config:VitsConfig):
        super().__init__(config)
        self.config = config
        self.model_config = config.model
        self.model_freezed = False

        self.generator = VitsModel(
            config=config,
            speaker_manager=self.speaker_manager,
            language_manager=self.language_manager,
            symbol_manager=self.symbol_manager
        )
        self.discriminator = VitsDiscriminator(
            periods=self.model_config.discriminator.periods_multi_period,
            use_spectral_norm=self.model_config.discriminator.use_spectral_norm,
        )

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

    def train_step(self, batch: Dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        spec_lens = batch["spec_lens"]
        if optimizer_idx == 0:
            spec = batch["spec"]
            waveform = batch["waveform"]
            speaker_ids = batch["speaker_ids"]
            speaker_embeds = batch["speaker_embeds"]
            language_ids = batch["language_ids"]

            # generator pass
            self.generator.train()
            outputs = self.generator.forward_vae(
                y=spec,
                y_lengths=spec_lens,
                waveform=waveform,
                speaker_embeds=speaker_embeds,
                speaker_ids=speaker_ids,
                language_ids=language_ids
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs

            # compute scores and features
            scores_disc_real, _, scores_disc_fake, _ = self.discriminator(
                x=outputs["waveform_seg"],
                x_hat=outputs["y_hat"].detach(),
            )

            # compute loss
            with autocast(enabled=False):
                loss_dict = criterion[0](
                    scores_disc_real=scores_disc_real,
                    scores_disc_fake=scores_disc_fake,
                )
            if self.model_freezed:
                return outputs, None
            else:
                return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel"]

            # compute melspec segment
            with autocast(enabled=False):
                mel_slice = segment(
                    x=mel.float(),
                    segment_indices=self.model_outputs_cache["slice_ids"],
                    segment_size=self.model_config.spec_segment_size,
                    pad_short=True
                )
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["y_hat"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            _, feats_disc_real, scores_disc_fake, feats_disc_fake = self.discriminator(
                x=self.model_outputs_cache["waveform_seg"],
                x_hat=self.model_outputs_cache["y_hat"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[1](
                    mel_slice=mel_slice_hat.float(),
                    mel_slice_hat=mel_slice.float(),
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in train_step()`"""
        return [VitsDiscriminatorLoss(self.config), VAEGeneratorLoss(self.config)]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters."""
        # select generator parameters
        discOptimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr[0],
            model=self.discriminator
        )

        genOptimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr[1],
            model=self.generator
        )
        return [discOptimizer, genOptimizer]

    def forward(self, input: torch.Tensor) -> Dict:
        print("nothing to do! doing the real train code in train_step. ")
        return input


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

    # for i in range(1, 10):
    #     train_model.test_run({"output_path": "/home/cano/output"})
    #     time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vits vctk pre-train VAE", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--config_path", type=str, default="./config/vits_vctk.json", required=False)
    args = parser.parse_args()

    # main(args.config_path)

    if platform.system() == "Windows":
        main("./config/vits_vctk.json")
    else:
        main("./config/vits_vctk_linux.json")