import math

import numpy as np
import torch
from config.config import VitsConfig
from language.language_manager import LanguageManager
from layers.duration_predictor import VitsDurationPredictor
from layers.encoder import TextEncoder, AudioEncoder
from layers.flow import ResidualCouplingBlocks
from layers.generator import HifiganGenerator
from layers.learnable_upsampling import LearnableUpsampling
from layers.quantizer import VAEMemoryBank
from util.monotonic_align import maximum_path
from speaker.speaker_manager import SpeakerManager
from text.symbol_manager import SymbolManager
from torch import nn
from torch.nn import functional as F
from util.helper import segment, rand_segments


class NaturalSpeechModel(nn.Module):
    def __init__(self, config:VitsConfig, speaker_manager:SpeakerManager, language_manager:LanguageManager, symbol_manager:SymbolManager):
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager
        self.symbol_manager = symbol_manager

        self.use_sdp = self.model_config.use_sdp
        self.spec_segment_size = self.model_config.spec_segment_size

        self.embedded_speaker_dim = self.model_config.speaker_embedding_channels
        if self.model_config.use_speaker_ids:
            self.speaker_embedding = nn.Embedding(self.speaker_manager.speaker_count(), self.embedded_speaker_dim)
        else:
            self.speaker_embedding = None
        self.embedded_language_dim = 0
        if self.model_config.use_language_ids:
            self.embedded_language_dim = self.model_config.language_embedding_channels
            self.language_embedding = nn.Embedding(self.language_manager.language_count(), self.embedded_language_dim)

        self.text_encoder = TextEncoder(
            n_vocab=self.symbol_manager.symbol_count(),
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
            in_channels=self.model_config.spec_channels,
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

    def align_duration(self, durations:torch.Tensor, target:torch.Tensor):
        """  align and extend the duration to the target duration length
        duration: [B,T]
        target: [B]
        """
        B = target.size(0)
        lengths = durations.sum(1)
        loss_duration_len = F.l1_loss(lengths, target)

        durations = durations * (target / lengths).unsqueeze(-1)
        durations = durations.cumsum(dim=1).round()
        durations = durations.diff(dim=1, prepend=torch.zeros([B,1]).to(durations.device))
        return durations, loss_duration_len

    def monotonic_align(self, z_p, m_p, logs_p, x, x_mask, y_mask):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            # code here does: using "Probability density function", by applying z_p to Probability density function
            # to get the probability
            o_scale = torch.exp(-2 * logs_p)  # 1/p**2, p is the Variance
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1] log( 1/sqrt(2*pi*p**2) )
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])  # log( -x**2/(2*p**2) ), x will be z_p here
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])  # log( 2xu/2p**2 ), u is the mean here
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]  log( u**2/(2*p**2) )
            logp = logp2 + logp3 + logp1 + logp4  # log("Probability density")
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        return attn_durations, attn

    def forward(
        self,
        x: torch.tensor,  # [B,C]
        x_lengths: torch.tensor,  # [B]
        y: torch.tensor,  # [B,specT,specC]
        y_lengths: torch.tensor,  # [B]
        waveform: torch.tensor,
        speaker_embeds = None,
        speaker_ids = None,
        language_ids = None,
        duration=None,
        use_gt_duration=True
    ):
        # speaker embedding
        g = None  # [b, h, 1]
        if self.model_config.use_speaker_embeds and speaker_embeds is not None:
            g = F.normalize(speaker_embeds).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)
        elif self.model_config.use_speaker_ids and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)

        # language embedding
        lang_embed = None
        if self.model_config.use_language_ids:
            lang_embed = self.language_embedding(language_ids).unsqueeze(-1)  # [B, lang_channel, 1]

        # audio encoder, encode audio to embedding layer z's dimension,
        z, m_q, logs_q, y_mask = self.audio_encoder(y, y_lengths, g=g)

        z_slice, ids_slice = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)
        if self.model_config.use_memory_bank:
            z_slice = self.memory_bank(z_slice)

        y_hat = self.waveform_decoder(z_slice, g=g)

        # backward flow layers
        z_p, total_logdet = self.flow(z, y_mask, g=g)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_embed)

        # gt_log_duration = torch.log(duration.unsqueeze(1) + 1e-6) * x_mask
        # differentiable durator (duration predictor & loss)
        pred_log_dur = self.duration_predictor(x, x_mask, g=g)  # pred_log_dur:[B,1,T]
        predict_durations = torch.exp(pred_log_dur) * x_mask  # w:[B,1,T]
        aligned_duration, loss_duration_len = self.align_duration(predict_durations.squeeze(1), y_mask.sum([1, 2]))  # aligned_duration: [B,T]

        # monotonic align and duration predictor
        attn_durations, attn = self.monotonic_align(z_p, m_p, logs_p, x, x_mask, y_mask)
        attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
        aligned_log_duration = torch.log(aligned_duration.unsqueeze(1) + 1e-6) * x_mask
        loss_duration = torch.sum((attn_log_durations - aligned_log_duration) ** 2, [1, 2]) / torch.sum(x_mask)

        # differentiable durator (learnable upsampling)
        upsampled_rep, p_mask, _, W = self.learnable_upsampling(
            duration=aligned_duration,
            tokens=x.transpose(1, 2),
            src_len=x_lengths,
            src_mask=~(x_mask.squeeze(1).bool()),
            max_src_len=x_lengths.max(),
        )
        m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)

        # forward flow layers
        z_q = self.flow(
            x=m_p + torch.randn_like(m_p) * torch.exp(logs_p),
            x_mask=p_mask.unsqueeze(1),
            g=g,
            reverse=True,
        )

        z_q_lengths = p_mask.flatten(1, -1).sum(dim=-1).long()
        z_slice_q, ids_slice_q = rand_segments( z_q, torch.minimum(z_q_lengths, y_lengths), self.spec_segment_size, let_short_samples=True, pad_short=True)
        if self.model_config.use_memory_bank:
            z_slice_q = self.memory_bank(z_slice_q)

        y_hat_e2e = self.waveform_decoder((z_slice_q), g=g)

        wav_seg = segment(
            x=waveform,
            segment_indices=ids_slice * self.config.audio.hop_length,
            segment_size=self.spec_segment_size * self.config.audio.hop_length,
            pad_short=True
        )
        wav_seg_e2e = segment(
            x=waveform,
            segment_indices=ids_slice_q * self.config.audio.hop_length,
            segment_size=self.spec_segment_size * self.config.audio.hop_length,
            pad_short=True
        )

        return {
            "y_hat": y_hat,  # the generated waveform
            "y_hat_e2e": y_hat_e2e,
            "ids_slice": ids_slice,
            "ids_slice_q": ids_slice_q,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "p_mask": p_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "z_q": z_q,
            "m_q": m_q,
            "logs_q": logs_q,
            "wav_seg": wav_seg,
            "wav_seg_e2e": wav_seg_e2e,
            "loss_duration_len": loss_duration_len,
            "loss_duration": loss_duration,
        }

    @torch.no_grad()
    def infer(
        self,
        x,  # [B, T_seq]
        x_lengths=None,  # [B]
        speaker_embeds=None,  # [T]
        speaker_ids=None,  # [B]
        language_ids=None,  # [B]
        noise_scale=1.0,
        length_scale=1.0,
        max_len=None
    ):
        if x_lengths is None:
            x_lengths = torch.LongTensor([len(a) for a in x])

        # speaker embedding
        g = None  # [b, h, 1]
        if self.model_config.use_speaker_embeds and speaker_embeds is not None:
            g = F.normalize(speaker_embeds).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)
        elif self.speaker_embedding is not None and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)

        # language embedding
        lang_embed = None
        if self.model_config.use_language_ids:
            lang_embed = self.language_embedding(language_ids).unsqueeze(-1)  # [B, lang_channel, 1]

        # infer with only one example
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_embed)

        logw = self.duration_predictor(x, x_mask, g=g, lang_emb=lang_embed)
        w = torch.exp(logw) * x_mask * length_scale
        duration = w.squeeze(1)

        upsampled_rep, p_mask, _, W = self.learnable_upsampling(
            duration=duration,
            tokens=x.transpose(1, 2),
            src_len=x_lengths,
            src_mask=~(x_mask.squeeze(1).bool()),
            max_src_len=x_mask.shape[-1]
        )
        m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)

        y_mask = p_mask.unsqueeze(1)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        if self.model_config.use_memory_bank:
            z = self.memory_bank(z)

        z = (z * y_mask)[:, :, :max_len]
        wav = self.waveform_decoder(z, g=g)
        return wav, duration, y_mask, (z, z_p, m_p, logs_p)

    @torch.no_grad()
    def generate_z_wav(self, spec, spec_len, speaker_id=None, speaker_embed=None):
        g = None  # [b, h, 1]
        if self.model_config.use_speaker_embeds and speaker_embed is not None:
            g = F.normalize(speaker_embed).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)
        elif self.speaker_embedding is not None and speaker_id is not None:
            g = self.speaker_embedding(speaker_id).unsqueeze(-1)
        z, _, _, _ = self.audio_encoder(spec, spec_len, g=g)
        wav = self.waveform_decoder(z, g=g)
        return wav

    def voice_conversion(self, y, y_lengths, source_speaker, target_speaker):
        """Forward pass for voice conversion
        Args:
            y (Tensor): Reference spectrograms. Tensor of shape [B, T, C]
            y_lengths (Tensor): Length of each reference spectrogram. Tensor of shape [B]
            source_speaker (Tensor): Reference speaker ID. Tensor of shape [B, T]
            target_speaker (Tensor): Target speaker ID. Tensor of shape [B, T]
        """

        # speaker embedding
        if self.model_config.use_speaker_ids:
            g_src = self.emb_g(torch.from_numpy((np.array(source_speaker))).unsqueeze(0)).unsqueeze(-1)
            g_tgt = self.emb_g(torch.from_numpy((np.array(target_speaker))).unsqueeze(0)).unsqueeze(-1)
        elif self.model_config.use_speaker_embeds:
            g_src = F.normalize(source_speaker).unsqueeze(-1)
            g_tgt = F.normalize(target_speaker).unsqueeze(-1)
        else:
            raise RuntimeError(" [!] Voice conversion is only supported on multi-speaker models.")

        z, _, _, y_mask = self.audio_encoder(y, y_lengths, g=g_src)
        z_p, _ = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
