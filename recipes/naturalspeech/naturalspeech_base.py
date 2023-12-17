from typing import Dict, Tuple, List
from torch.cuda.amp import autocast
from config.config import VitsConfig
from layers.discriminator import VitsDiscriminator
from layers.losses import NaturalSpeechDiscriminatorLoss, NaturalSpeechGeneratorLoss
from recipes.naturalspeech.naturalspeech import NaturalSpeechModel
from recipes.trainer_model import TrainerModelWithDataset
from text import text_to_tokens, _intersperse
from torch import nn
from trainer import torch, get_optimizer
from util.helper import segment
from util.mel_processing import wav_to_mel


class NaturalSpeechBase(TrainerModelWithDataset):
    """
    Natural Speech model training model.
    """
    def __init__(self, config:VitsConfig):
        super().__init__(config)
        self.config = config
        self.model_config = config.model

        self.generator = NaturalSpeechModel(config=config, speaker_manager=self.speaker_manager, language_manager=self.language_manager)
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
            language_ids = batch["language_ids"]

            # generator pass
            self.generator.train()
            outputs = self.generator(
                x=tokens,
                x_lengths=token_lens,
                y=spec,
                y_lengths=spec_lens,
                waveform=waveform,
                speaker_embeds=speaker_embeds,
                speaker_ids=speaker_ids,
                language_ids=language_ids,
                duration=None,
                use_gt_duration=self.model_config.use_gt_duration
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs

            # compute scores and features
            scores_disc_real, _, scores_disc_fake, _ = self.discriminator(
                x=outputs["wav_seg"],
                x_hat=outputs["y_hat"].detach()
            )
            scores_disc_real_e2e, _, scores_disc_fake_e2e, _ = self.discriminator(
                x=outputs["wav_seg_e2e"],
                x_hat=outputs["y_hat_e2e"].detach()
            )

            # compute loss
            with autocast(enabled=False):
                loss_dict = criterion[0](
                    scores_disc_real=scores_disc_real,
                    scores_disc_fake=scores_disc_fake,
                )
                loss_dict_e2e = criterion[0](
                    scores_disc_real=scores_disc_real_e2e,
                    scores_disc_fake=scores_disc_fake_e2e,
                )
                loss_disc_all = loss_dict_e2e
                loss_disc_all["loss"] = loss_dict["loss"] + loss_dict_e2e["loss"]
            return outputs, loss_disc_all

        if optimizer_idx == 1:
            mel = batch["mel"]

            # compute melspec segment
            with autocast(enabled=False):
                mel_slice = segment(
                    x=mel.float(),
                    segment_indices=self.model_outputs_cache["ids_slice"],
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
            scores_disc_real, feats_disc_real, scores_disc_fake, feats_disc_fake = self.discriminator(
                x=self.model_outputs_cache["wav_seg"],
                x_hat=self.model_outputs_cache["y_hat"]
            )
            scores_disc_real_e2e, _, scores_disc_fake_e2e, _ = self.discriminator(
                x=self.model_outputs_cache["wav_seg_e2e"],
                x_hat=self.model_outputs_cache["y_hat_e2e"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[1](
                    mel_slice=mel_slice,
                    mel_slice_hat=mel_slice_hat,
                    scores_disc_fake=scores_disc_fake,
                    scores_disc_fake_e2e=scores_disc_fake_e2e,
                    feats_disc_real=feats_disc_real,
                    feats_disc_fake=feats_disc_fake,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    loss_pitch = None,
                    z_p=self.model_outputs_cache["z_p"],
                    m_p=self.model_outputs_cache["m_p"],
                    logs_p=self.model_outputs_cache["logs_p"],
                    z_q=self.model_outputs_cache["z_q"],
                    m_q=self.model_outputs_cache["m_q"],
                    logs_q=self.model_outputs_cache["logs_q"],
                    p_mask=self.model_outputs_cache["p_mask"],
                    z_mask=self.model_outputs_cache["y_mask"],
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    @torch.no_grad()
    def inference(self, text:str, speaker_name:str=None, speaker_embed=None, language:str=None):
        lang = "en" if language is None else language
        tokens = text_to_tokens(text, cleaner_name=self.config.text.text_cleaners.get(lang))
        if self.config.text.add_blank:
            tokens = _intersperse(tokens, 0)
        tokens = torch.LongTensor(tokens).unsqueeze(dim=0).cuda()
        x_lengths = torch.LongTensor([tokens.size(1)]).cuda()

        speaker_ids = None
        if speaker_name is not None:
            speaker_id = self.speaker_manager.get_speaker_id(speaker_name)
            speaker_ids = torch.LongTensor([speaker_id]).cuda()
        speaker_embeds = None
        if speaker_embed is not None:
            speaker_embeds = torch.FloatTensor([speaker_embed]).cuda()
        language_ids = None
        if language is not None:
            language_id = self.language_manager.get_language_id(language)
            language_ids = torch.LongTensor([language_id]).cuda()

        self.generator.eval()
        wav, _, _, _ = self.generator.infer(
            tokens,
            x_lengths,
            speaker_ids=speaker_ids,
            speaker_embeds=speaker_embeds,
            language_ids=language_ids,
            noise_scale=0.667,
            length_scale=1.0
        )
        return wav

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in train_step()`"""
        return [NaturalSpeechDiscriminatorLoss(self.config), NaturalSpeechGeneratorLoss(self.config)]

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

