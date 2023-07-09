from itertools import chain
from typing import Dict, List, Union, Tuple

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from trainer import TrainerModel
from trainer.trainer_utils import get_optimizer, get_scheduler
from config.config import TrainTTSConfig
from dataset.dataset import TextAudioDataset, DistributedBucketSampler, TextAudioCollate
from layers.discriminator import VitsDiscriminator
from layers.losses import VitsDiscriminatorLoss, VitsGeneratorLoss
from layers.encoder import TextEncoder, AudioEncoder
from util.helper import sequence_mask
from util.mel_processing import wav_to_spec, spec_to_mel


class SpeechModel(TrainerModel):
    def __init__(self,config:TrainTTSConfig):
        super().__init__()
        self.config = config
        self.model_config = config.model

        self.text_encoder = TextEncoder(
            n_vocab=self.model_config.text_encoder.num_chars,
            out_channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            hidden_channels_ffn=self.model_config.text_encoder.hidden_channels_ffn_text_encoder,
            num_heads=self.model_config.text_encoder.num_heads_text_encoder,
            num_layers=self.model_config.text_encoder.num_layers_text_encoder,
            kernel_size=self.model_config.text_encoder.kernel_size_text_encoder,
            dropout_p=self.model_config.text_encoder.dropout_p_text_encoder,
            language_emb_dim=self.model_config.embedded_language_dim,
        )

        self.audio_encoder = AudioEncoder(
            self.model_config.out_channels,
            self.model_config.hidden_channels,
            self.model_config.hidden_channels,
            kernel_size=self.model_config.audio_encoder.kernel_size_audio_encoder,
            dilation_rate=self.model_config.audio_encoder.dilation_rate_audio_encoder,
            num_layers=self.model_config.audio_encoder.num_layers_audio_encoder,
            cond_channels=self.model_config.embedded_speaker_dim,
        )

        self.discriminator = VitsDiscriminator(
            periods=self.model_config.discriminator.periods_multi_period_discriminator,
            use_spectral_norm=self.model_config.discriminator.use_spectral_norm_disriminator,
        )

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
            return loader

        dataset = TextAudioDataset(samples, config.dataset_config)

        # wait all the DDP process to be ready
        if num_gpus > 1:
            dist.barrier()

        # get samplers and collate
        sampler = DistributedBucketSampler(
            dataset,
            config.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=num_gpus,
            rank=rank,
            shuffle=True
        )
        collate_fn = TextAudioCollate()

        if is_eval:
            loader = DataLoader(
                dataset,
                num_workers=config.num_eval_loader_workers,
                shuffle=False,
                pin_memory=False,
                collate_fn=collate_fn,
                batch_size=config.eval_batch_size,
            )
        else:
            loader = DataLoader(
                dataset,
                num_workers=config.batch_size,
                shuffle=False,
                pin_memory=False,
                collate_fn=collate_fn,
                batch_sampler=sampler,
            )
        return loader

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        wav = batch["waveform"]
        dataset_config = self.config.dataset_config
        spec = wav_to_spec(
            wav=wav,
            n_fft=dataset_config.fft_length,
            hop_size=dataset_config.hop_length,
            win_size=dataset_config.win_length,
            center=False
        )
        batch["spec"] = spec

        mel = spec_to_mel(
            spec=spec,
            n_fft=dataset_config.fft_length,
            num_mels=dataset_config.num_mels,
            sample_rate=dataset_config.sample_rate,
            fmin=dataset_config.mel_fmin,
            fmax=dataset_config.mel_fmax
        )
        batch["mel"] = mel

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

    def forward(self, input: torch.Tensor, *args, aux_input={}, **kwargs) -> Dict:
        x = input
        return x


    def train_step(self, batch: Dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        print("batch size: %i" % len(batch))
        if optimizer_idx == 0:
            output = {}
            total_loss = {}
            return output, total_loss

        if optimizer_idx == 1:
            output = {}
            total_loss = {}
            return output, total_loss

        raise ValueError(" [!] Unexpected `optimizer_idx`.")


    def eval_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        pass


    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in train_step()`"""
        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        # select generator parameters
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr, self.discriminator)

        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("discriminator."))
        print([k for k, params in self.named_parameters()])
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.
        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.
        Returns:
            List: Schedulers, one for each optimizer.
        """
        lr_scheduler_params = {
            "gamma": 0.999875,
            "last_epoch": -1
        }
        scheduler_D = get_scheduler("ExponentialLR", lr_scheduler_params, optimizer[0])
        scheduler_G = get_scheduler("ExponentialLR", lr_scheduler_params, optimizer[1])
        return [scheduler_D, scheduler_G]