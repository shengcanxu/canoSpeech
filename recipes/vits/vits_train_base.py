from typing import Dict, Tuple, List
import torch
from config.config import VitsConfig
from coqpit import Coqpit
from dataset.basic_dataset import TextAudioDataset
from layers.discriminator import VitsDiscriminator
from layers.losses import VitsDiscriminatorLoss, VitsGeneratorLoss
from recipes.trainer_model import TrainerModelWithDataset
from recipes.vits.vits import VitsModel
from text import _intersperse
from torch import nn
from torch.cuda.amp import autocast
from trainer import get_optimizer
from util.helper import segment
from util.mel_processing import wav_to_mel, wav_to_spec


class VitsTrainBase(TrainerModelWithDataset):
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

    def _freeze_layers(self):
        for param in self.generator.text_encoder.parameters():
            param.requires_grad = False
        for param in self.generator.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.generator.flow.parameters():
            param.requires_grad = False
        for param in self.generator.duration_predictor.parameters():
            param.requires_grad = False
        for param in self.generator.waveform_decoder.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = False
        self.model_freezed = True

    # def on_init_end(self, trainer) -> None:
    #     print("freeze the layers...")
    #     self._freeze_layers()

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
                    m_p_dur=self.model_outputs_cache["m_p_dur"].float(),
                    logs_p_dur=self.model_outputs_cache["logs_p_dur"].float(),
                    z_q_dur=self.model_outputs_cache["z_q_dur"].float(),
                    m_q_audio=self.model_outputs_cache["m_q_audio"].float(),
                    logs_q_audio=self.model_outputs_cache["logs_q_audio"].float(),
                    spec_lens=spec_lens,
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    total_logdet=self.model_outputs_cache["total_logdet"],
                    use_speaker_encoder_as_loss=self.model_config.use_speaker_encoder_as_loss,
                    gt_speaker_emb=self.model_outputs_cache["gt_speaker_emb"],
                    syn_speaker_emb=self.model_outputs_cache["syn_speaker_emb"],
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    @torch.no_grad()
    def inference(self, text:str, speaker_name:str=None, speaker_embed=None, ref_spec=None, language:str=None):
        lang = "en" if language is None else language
        tokens = self.symbol_manager.text_to_tokens(text, cleaner_name=self.config.text.text_cleaners.get(lang), lang=lang)
        if self.config.text.add_blank:
            tokens = _intersperse(tokens, 0)
        tokens = torch.LongTensor(tokens).unsqueeze(dim=0).cuda()
        x_lengths = torch.LongTensor([tokens.size(1)]).cuda()

        speaker_ids = None
        if speaker_name is not None:
            speaker_id = self.speaker_manager.get_speaker_id(speaker_name)
            speaker_ids = torch.LongTensor([speaker_id]).cuda()
        speaker_embeds = None
        if speaker_embed is not None and self.model_config.use_speaker_embeds:
            speaker_embeds = torch.FloatTensor([speaker_embed]).cuda()
        language_ids = None
        if language is not None and self.model_config.use_speaker_ids:
            language_id = self.language_manager.get_language_id(language)
            language_ids = torch.LongTensor([language_id]).cuda()

        self.generator.eval()
        wav, _, _, _ = self.generator.infer(
            tokens,
            x_lengths,
            speaker_ids=speaker_ids,
            speaker_embeds=speaker_embeds,
            ref_spec=ref_spec,
            language_ids = language_ids,
            noise_scale=0.667,
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

    def forward(self, input: torch.Tensor) -> Dict:
        print("nothing to do! doing the real train code in train_step. ")
        return input
