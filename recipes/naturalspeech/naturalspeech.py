import time

import torch
from coqpit import Coqpit
from typing import Dict, List, Union, Tuple

from torch import nn
from torch.cuda.amp import autocast
from trainer import get_optimizer, get_scheduler
from torch.nn import functional as F
from config.config import NaturalSpeechConfig
from language.languages import LanguageManager
from layers.discriminator import VitsDiscriminator
from layers.duration_predictor import VitsDurationPredictor
from layers.encoder import TextEncoder, AudioEncoder
from layers.flow import ResidualCouplingBlocks
from layers.generator import HifiganGenerator
from layers.learnable_upsampling import LearnableUpsampling
from layers.losses import NaturalSpeechDiscriminatorLoss, NaturalSpeechGeneratorLoss
from layers.quantizer import VAEMemoryBank
from recipes.trainer_model import TrainerModelWithDataset
from text import text_to_tokens
from util.helper import segment, rand_segments
from util.mel_processing import wav_to_mel


class NaturalSpeechModel(nn.Module):
    def __init__(self, config:NaturalSpeechConfig, speaker_embed: torch.Tensor = None, language_manager: LanguageManager = None ):
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.speaker_embed = speaker_embed
        self.language_manager = language_manager

        # init multi-speaker, speaker_embedding is used when the speaker_embed is not provided
        self.num_speakers = self.model_config.num_speakers
        self.spec_segment_size = self.model_config.spec_segment_size
        self.embedded_speaker_dim = self.model_config.speaker_embedding_channels
        if self.num_speakers > 0:
            self.speaker_embedding = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

        self.init_multilingual(config)

        self.text_encoder = TextEncoder(
            n_vocab=self.model_config.text_encoder.num_chars,
            out_channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            hidden_channels_ffn=self.model_config.text_encoder.hidden_channels_ffn,
            num_heads=self.model_config.text_encoder.num_heads,
            num_layers=self.model_config.text_encoder.num_layers,
            kernel_size=self.model_config.text_encoder.kernel_size,
            dropout_p=self.model_config.text_encoder.dropout_p,
            language_emb_dim=self.embedded_language_dim,
        )
        self.audio_encoder = AudioEncoder(
            in_channels=self.model_config.out_channels,
            out_channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.audio_encoder.kernel_size,
            dilation_rate=self.model_config.audio_encoder.dilation_rate,
            num_layers=self.model_config.audio_encoder.num_layers,
            cond_channels=self.embedded_speaker_dim,
        )
        self.flow = ResidualCouplingBlocks(
            channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.flow.kernel_size,
            dilation_rate=self.model_config.flow.dilation_rate,
            num_flows=self.model_config.flow.num_flows,
            num_layers=self.model_config.flow.num_layers_in_flow,
            cond_channels=self.embedded_speaker_dim,
        )
        self.duration_predictor = VitsDurationPredictor(
            in_channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.duration_predictor.filter_channels,
            kernel_size=self.model_config.duration_predictor.kernel_size,
            dropout_p=self.model_config.duration_predictor.dropout_p,
            cond_channels=self.embedded_speaker_dim,
            language_emb_dim=self.embedded_language_dim,
        )
        self.learnable_upsampling = LearnableUpsampling(
            d_predictor=self.model_config.learnable_upsampling.d_predictor,
            kernel_size=self.model_config.learnable_upsampling.kernel_size_lu,
            dropout=self.model_config.learnable_upsampling.dropout_lu,
            conv_output_size=self.model_config.learnable_upsampling.conv_output_size,
            dim_w=self.model_config.learnable_upsampling.dim_w,
            dim_c=self.model_config.learnable_upsampling.dim_c,
            max_seq_len=self.model_config.learnable_upsampling.max_seq_len
        )
        self.waveform_decoder = HifiganGenerator(
            in_channels=self.model_config.hidden_channels,
            out_channels=1,
            resblock_type=self.model_config.waveform_decoder.resblock_type,
            resblock_dilation_sizes=self.model_config.waveform_decoder.resblock_dilation_sizes,
            resblock_kernel_sizes=self.model_config.waveform_decoder.resblock_kernel_sizes,
            upsample_kernel_sizes=self.model_config.waveform_decoder.upsample_kernel_sizes,
            upsample_initial_channel=self.model_config.waveform_decoder.upsample_initial_channel,
            upsample_factors=self.model_config.waveform_decoder.upsample_rates,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False
        )
        self.memory_bank = VAEMemoryBank(
            bank_size=self.model_config.memory_bank.bank_size,
            n_hidden_dims=self.model_config.memory_bank.n_hidden_dims,
            n_attn_heads=self.model_config.memory_bank.n_attn_heads
        )

    def init_multilingual(self, config: Coqpit):
        """Initialize multilingual modules of a model.
        Args:
            config (Coqpit): Model configuration.
        """
        if self.model_config.language_ids_file is not None:
            self.language_manager = LanguageManager(language_ids_file_path=config.language_ids_file)

        if self.model_config.use_language_embedding and self.language_manager:
            print(" > initialization of language-embedding layers.")
            self.num_languages = self.language_manager.num_languages
            self.embedded_language_dim = self.model_config.language_embedding_channels
            self.language_embedding = nn.Embedding(self.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.language_embedding.weight)
        else:
            self.embedded_language_dim = 0

    @staticmethod
    def _set_cond_input(speaker_embeds, speaker_ids, language_ids):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid, durations = None, None, None, None
        if speaker_ids is not None:
            sid = speaker_ids
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)
        if speaker_embeds is not None:
            g = F.normalize(speaker_embeds).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if language_ids is not None:
            lid = language_ids
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)
        return sid, g, lid

    def forward(
        self,
        x,  # [B,C]
        x_lengths,  # [B]
        y,  # [B,specT,specC]
        y_lengths,  # [B]
        duration=None,
        use_gt_duration=True,
        speaker_embeds=None,
        language_ids=None
    ):
        # _, g, lid = self._set_cond_input(speaker_embeds, None, language_ids)
        if speaker_embeds is not None:
            g = F.normalize(speaker_embeds).unsqueeze(-1)
        else:
            g = torch.zeros(x.size(0), self.embedded_speaker_dim, 1).to(x.device)
        if g.ndim == 2:
            g = g.unsqueeze_(0)
        lid = None

        # language embedding
        lang_embedding = None
        if self.model_config.use_language_embedding and lid is not None:
            lang_embedding = self.language_embedding(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_embedding)

        # audio encoder, encode audio to embedding layer z's dimension,
        # z:[B,C,spectrogramToken] m_q:[B,C,specT] logs_q:[B,C,specT] y_mask:[B,1,specT]
        z, m_q, logs_q, y_mask = self.audio_encoder(y, y_lengths, g=g)

        # z_p:[B,C,specT]
        z_p = self.flow(z, y_mask, g=g)

        # differentiable durator (duration predictor & loss)
        logw = self.duration_predictor(x, x_mask, g=g)  # logw:[B,1,T]
        w = torch.exp(logw) * x_mask  # w:[B,1,T]

        w_gt = duration.unsqueeze(1)
        logw_gt = torch.log(w_gt + 1e-6) * x_mask
        # for averaging
        duration_loss = torch.sum((logw - logw_gt) ** 2, [1, 2]) / torch.sum(x_mask)
        # use predicted duration
        if not use_gt_duration:
            duration = w.squeeze(1)

        # differentiable durator (learnable upsampling)
        upsampled_rep, p_mask, _, W = self.learnable_upsampling(
            duration=duration,
            V=x.transpose(1, 2),
            src_len=x_lengths,
            src_mask=~(x_mask.squeeze(1).bool()),
            tgt_len=y_lengths,
            max_src_len=x_lengths.max(),
        )
        p_mask = ~p_mask

        m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)
        z_slice, ids_slice = rand_segments(
            z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True
        )

        if self.model_config.use_memory_bank:
            z_slice = self.memory_bank(z_slice)

        y_hat = self.waveform_decoder(z_slice, g=g)

        z_q = self.flow(
            x=m_p + torch.randn_like(m_p) * torch.exp(logs_p),
            x_mask=p_mask.unsqueeze(1),
            g=g,
            reverse=True,
        )
        z_q_lengths = p_mask.flatten(1, -1).sum(dim=-1).long()
        z_slice_q, ids_slice_q = rand_segments(
            z_q, torch.minimum(z_q_lengths, y_lengths), self.spec_segment_size, let_short_samples=True, pad_short=True
        )

        if self.model_config.use_memory_bank:
            z_slice_q = self.memory_bank(z_slice_q)

        y_hat_e2e = self.waveform_decoder((z_slice_q), g=g)

        return {
            "y_hat": y_hat,  # the generated waveform
            "duration_loss": duration_loss,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
            "p_mask": p_mask,
            "W": W,
            "y_hat_e2e": y_hat_e2e,
            "z_q": z_q,
            "duration": duration,
            "ids_slice_q": ids_slice_q
        }

    @torch.no_grad()
    def infer(
        self,
        x,  # [B, T_seq]
        x_lengths=None,  # [B]
        duration=None,
        speaker_embeds=None,  # [B]
        language_ids=None  # [B]
    ):
        if x_lengths is None:
            x_lengths = torch.LongTensor([len(a) for a in x])

        if speaker_embeds is not None:
            g = F.normalize(speaker_embeds).unsqueeze(-1)
        else:
            g = torch.zeros(x.size(0), self.embedded_speaker_dim, 1).to(x.device)
        if g.ndim == 2:
            g = g.unsqueeze_(0)
        lid = None

        # language embedding
        lang_embedding = None
        if self.model_config.use_language_embedding and lid is not None:
            lang_embedding = self.language_embedding(lid).unsqueeze(-1)

        # infer with only one example
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_embedding)

        logw = self.duration_predictor(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * self.model_config.length_scale
        if duration is not None:
            w = duration.unsqueeze(1) * x_mask * self.model_config.length_scale

        upsampled_rep, p_mask, _, W = self.learnable_upsampling(
            duration=w.squeeze(1),
            V=x.transpose(1, 2),
            src_len=x_lengths,
            src_mask=~(x_mask.squeeze(1).bool()),
            tgt_len=None,
            max_src_len=x_mask.shape[-1]
        )
        p_mask = ~p_mask
        m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)

        y_mask = p_mask.unsqueeze(1)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.model_config.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        if self.model_config.use_memory_bank:
            z = self.memory_bank(z)

        z = (z * y_mask)[:, :, :self.model_config.max_inference_len]
        y_hat = self.waveform_decoder(z, g=g)
        return y_hat


class NaturalSpeechTrain(TrainerModelWithDataset):
    """
    Natural Speech model training model.
    """
    def __init__(self, config:NaturalSpeechConfig, speaker_embed: torch.Tensor = None, language_manager: LanguageManager = None, ):
        super().__init__()
        self.config = config
        self.model_config = config.model

        self.generator = NaturalSpeechModel(
            config=config,
            speaker_embed=speaker_embed,
            language_manager=language_manager
        )
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
            speaker_embeds = batch["speaker_embed"]
            wav = batch["waveform"]
            duration = batch["duration"]

            # generator pass
            outputs = self.generator(
                x=tokens,
                x_lengths=token_lens,
                y=spec,
                y_lengths=spec_lens,
                duration=duration,
                use_gt_duration=self.model_config.use_gt_duration,
                speaker_embeds=speaker_embeds,
                language_ids=None,

            )

            wav_slice1 = segment(
                x=wav,
                segment_indices=outputs["ids_slice"] * self.config.audio.hop_length,
                segment_size=self.model_config.spec_segment_size * self.config.audio.hop_length,
                pad_short=True
            )
            wav_slice2 = segment(
                x=wav,
                segment_indices=outputs["ids_slice_q"] * self.config.audio.hop_length,
                segment_size=self.model_config.spec_segment_size * self.config.audio.hop_length,
                pad_short=True
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs
            self.wav_slice1 = wav_slice1
            self.wav_slice2 = wav_slice2

            # compute scores and features
            scores_disc_real, _, scores_disc_fake, _ = self.discriminator(
                x=wav_slice1,
                x_hat=outputs["y_hat"].detach()
            )
            scores_disc_real_e2e, _, scores_disc_fake_e2e, _ = self.discriminator(
                x=wav_slice2,
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
                loss_disc_all = loss_dict["loss"] + loss_dict_e2e["loss"]
                loss_dict_e2e["loss"] = loss_disc_all
            return outputs, loss_dict_e2e

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
                x=self.wav_slice1,
                x_hat=self.model_outputs_cache["y_hat"]
            )
            scores_disc_real_e2e, _, scores_disc_fake_e2e, _ = self.discriminator(
                x=self.wav_slice2,
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
                    loss_duration_length=self.model_outputs_cache["duration_loss"],
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
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in train_step()`"""
        return [NaturalSpeechDiscriminatorLoss(self.config), NaturalSpeechGeneratorLoss(self.config)]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters."""
        # select generator parameters
        disOptimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr,
            model=self.discriminator
        )

        # gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("discriminator."))
        genOptimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr,
            model=self.generator
        )
        return [disOptimizer, genOptimizer]

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

    def forward(self, input: torch.Tensor) -> Dict:
        print("nothing to do! doing the real train code in train_step. ")
        return input

    def infer(self, text:str):
        tokens = text_to_tokens(text)
        tokens = torch.LongTensor(tokens).unsqueeze(dim=0)
        wav = self.generator.infer(tokens)
        return wav