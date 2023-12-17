import math
import os
from typing import Dict
import numpy as np
import torch
import torchaudio
from language.language_manager import LanguageManager
from speaker.speaker_manager import SpeakerManager
from torch.nn import functional as F
from torch import nn
from config.config import VitsConfig
from layers.duration_predictor import VitsDurationPredictor, generate_path, StochasticDurationPredictor
from layers.flow import ResidualCouplingBlocks
from layers.generator import HifiganGenerator
from layers.encoder import TextEncoder, AudioEncoder
from monotonic_align.maximum_path import maximum_path
from speaker.speaker_encoder import SpeakerEncoder
from util.helper import sequence_mask, segment, rand_segments


# VITS model and yourTTS model 
class VitsModel(nn.Module):
    def __init__(self, config:VitsConfig, speaker_manager:SpeakerManager, language_manager:LanguageManager):
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager

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

        if self.model_config.use_speaker_encoder_as_loss:
            self.speaker_encoder = SpeakerEncoder(
                config_path=os.path.dirname(__file__) + "/../../speaker/speaker_encoder_config.json",
                model_path=os.path.dirname(__file__) + "/../../speaker/speaker_encoder_model.pth.tar",
                use_cuda=torch.cuda.is_available(),
            )
            self.audio_transform = torchaudio.transforms.Resample(
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.speaker_encoder.encoder.audio_config["sample_rate"],
            )

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
        if self.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                in_channels=self.model_config.hidden_channels,
                hidden_channels=self.model_config.hidden_channels,
                kernel_size=self.model_config.duration_predictor.kernel_size,
                dropout_p=self.model_config.duration_predictor.dropout_p,
                num_flows=4,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
            )
        else:
            self.duration_predictor = VitsDurationPredictor(
                in_channels=self.model_config.hidden_channels,
                hidden_channels=256,
                kernel_size=self.model_config.duration_predictor.kernel_size,
                dropout_p=self.model_config.duration_predictor.dropout_p,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
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
            conv_pre_weight_norm=True,
            conv_post_weight_norm=True,
            conv_post_bias=False,
        )

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
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        speaker_embeds = None,
        speaker_ids = None,
        language_ids = None
    ) -> Dict:
        """Forward pass of the model.
        Args:
            x (torch.tensor): Batch of input character sequence IDs. [B, T_seq]`
            x_lengths (torch.tensor): Batch of input character sequence lengths. [B]`
            y (torch.tensor): Batch of input spectrograms. [B, C, T_spec]`
            y_lengths (torch.tensor): Batch of input spectrogram lengths. [B]`
            waveform (torch.tensor): Batch of ground truth waveforms per sample. [B, 1, T_wav]`
            speaker_embeds:[B, C, 1]: Batch of speaker embedding
            speaker_ids:`[B]: Batch of speaker ids. use_speaker_ids should be true
            language_ids:`[B]: Batch of language ids.
        """
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

        # audio encoder, encode audio to embedding layer z's dimension
        z, m_q, logs_q, y_mask = self.audio_encoder(y, y_lengths, g=g)

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        y_hat = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            x=waveform,
            segment_indices=slice_ids * self.config.audio.hop_length,
            segment_size=self.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_embed)

        # monotonic align and duration predictor
        attn_durations, attn = self.monotonic_align(z_p, m_p, logs_p, x, x_mask, y_mask)
        # expand text token_size to audio token_size
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        if self.use_sdp:
            loss_duration = self.duration_predictor(
                x=x.detach(),
                x_mask=x_mask,
                dr=attn_durations,
                g=g.detach() if g is not None else g,
                lang_emb=lang_embed,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            predict_log_durations = self.duration_predictor(
                x=x.detach(),
                x_mask=x_mask,
                g=g.detach() if g is not None else g,
                lang_emb=lang_embed,
            )
            loss_duration = torch.sum((predict_log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
            # loss_duration = torch.sum((torch.exp(predict_log_durations) - attn_durations) ** 2, [1, 2]) / torch.sum(x_mask)

        if self.model_config.use_speaker_encoder_as_loss and self.speaker_encoder.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, y_hat), dim=0)

            # resample audio to speaker encoder sample_rate
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_encoder.encoder.forward(wavs_batch, l2_norm=True)
            # split generated and GT speaker embeddings
            gt_speaker_emb, syn_speaker_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_speaker_emb, syn_speaker_emb = None, None

        return {
            "y_hat": y_hat,  # [B, 1, T_wav]
            "m_p": m_p,  # [B, C, T_dec]
            "logs_p": logs_p,  # [B, C, T_dec]
            "z": z,  # [B, C, T_dec]
            "z_p": z_p,  # [B, C, T_dec]
            "m_q": m_q,  # [B, C, T_dec]
            "logs_q": logs_q,  # [B, C, T_dec]
            "waveform_seg": wav_seg,  # [B, 1, spec_seg_size * hop_length]
            "slice_ids": slice_ids,
            "loss_duration": loss_duration,
            "gt_speaker_emb": gt_speaker_emb,
            "syn_speaker_emb": syn_speaker_emb,
        }

    @torch.no_grad()
    def infer(
        self,
        x,  # [B, T_seq]
        x_lengths,  # [B]
        speaker_embeds=None,  # [T]
        speaker_ids=None,  # [B]
        language_ids=None,  # [B]
        noise_scale=1.0,
        length_scale=1.0,
        max_len=None
    ):
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

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_embed)

        if self.use_sdp:
            logw = self.duration_predictor(
                x=x,
                x_mask=x_mask,
                g=g if g is not None else g,
                reverse=True,
                noise_scale=noise_scale,
            )
        else:
            logw = self.duration_predictor(
                x=x,
                x_mask=x_mask,
                g=g if g is not None else g,
                lang_emb=lang_embed
            )
        w = torch.exp(logw) * x_mask * length_scale

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        z = (z * y_mask)[:, :, : max_len]
        wav = self.waveform_decoder(z, g=g)
        return wav, attn, y_mask, (z, z_p, m_p, logs_p)

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
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
