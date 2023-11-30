from typing import Dict, Tuple, List
import torch
from config.config import VitsConfig
from coqpit import Coqpit
from dataset.basic_dataset import TextAudioDataset
from layers.discriminator import VitsDiscriminator
from layers.losses import VitsDiscriminatorLoss, VitsGeneratorLoss
from recipes.trainer_model import TrainerModelWithDataset
from recipes.vits.vits import VitsModel
from torch import nn
from torch.cuda.amp import autocast
from trainer import get_optimizer
from util.helper import segment
from util.mel_processing import wav_to_mel


class VitsTrain_Base(TrainerModelWithDataset):
    """
    VITS and YourTTS model training model.
    """
    def __init__(self, config:VitsConfig):
        super().__init__(config)
        self.config = config
        self.model_config = config.model
        self.skip_discriminator = False

        self.generator = VitsModel(config=config)
        self.discriminator = VitsDiscriminator(
            periods=self.model_config.discriminator.periods_multi_period,
            use_spectral_norm=self.model_config.discriminator.use_spectral_norm,
        )

    def train_step(self, batch: Dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        spec_lens = batch["spec_lens"]
        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lens = batch["token_lens"]
            spec = batch["spec"]
            waveform = batch["waveform"]
            speaker_ids = batch["speaker_ids"]
            speaker_embeds = batch["speaker_embeds"]

            # generator pass
            outputs = self.generator(
                x=tokens,
                x_lengths=token_lens,
                y=spec,
                y_lengths=spec_lens,
                waveform=waveform,
                speaker_embeds=speaker_embeds,
                speaker_ids=speaker_ids,
                language_ids=None
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

            self.disc_loss_dict = loss_dict
            if self.skip_discriminator:
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
                    z_p=self.model_outputs_cache["z_p"].float(),
                    logs_q=self.model_outputs_cache["logs_q"].float(),
                    m_p=self.model_outputs_cache["m_p"].float(),
                    logs_p=self.model_outputs_cache["logs_p"].float(),
                    z_len=spec_lens,
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    use_speaker_encoder_as_loss=self.model_config.use_speaker_encoder_as_loss,
                    gt_speaker_emb=None,
                    syn_speaker_emb=None,
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in train_step()`"""
        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

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

    def get_dataset(self, config: Coqpit, samples):
        return TextAudioDataset(samples, config)

    def forward(self, input: torch.Tensor) -> Dict:
        print("nothing to do! doing the real train code in train_step. ")
        return input

