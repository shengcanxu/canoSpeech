import math
import os
from typing import Dict
import numpy as np
import torch
import torchaudio
from language.language_manager import LanguageManager
from speaker.speaker_manager import SpeakerManager
from text.symbol_manager import SymbolManager
from torch.nn import functional as F
from torch import nn
from config.config import VitsConfig
from layers.duration_predictor import VitsDurationPredictor, generate_path, StochasticDurationPredictor
from layers.flow import ResidualCouplingBlocks
from layers.generator import HifiganGenerator
from layers.encoder import TextEncoder, AudioEncoder, ReferenceEncoder
from monotonic_align.maximum_path import maximum_path
from speaker.speaker_encoder import SpeakerEncoder
from util.helper import sequence_mask, segment, rand_segments


# VITS model and yourTTS model 
class VitsModel(nn.Module):
    def __init__(self, config:VitsConfig, speaker_manager:SpeakerManager, language_manager:LanguageManager, symbol_manager:SymbolManager):
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager
        self.symbol_manager = symbol_manager

        self.use_sdp = self.model_config.use_sdp
        self.use_SNAC = self.model_config.flow.use_SNAC
        self.spec_segment_size = self.model_config.spec_segment_size
        self.mas_noise_scale = self.model_config.mas_noise_scale or 0.0
        self.mas_noise_scale_decay = self.model_config.mas_noise_scale_decay or 0.0

        self.embedded_speaker_channels = self.model_config.speaker_embedding_channels
        if self.model_config.use_speaker_ids:
            self.speaker_embedding = nn.Embedding(self.speaker_manager.speaker_count(), self.embedded_speaker_channels)
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
        elif self.model_config.use_speaker_encoder:
            self.speaker_encoder = ReferenceEncoder(
                spec_channels=self.model_config.spec_channels,
                gin_channels=self.model_config.speaker_embedding_channels
            )

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
            speaker_emb_dim=self.embedded_speaker_channels if self.model_config.text_encoder.use_speaker_embed else 0,
        )
        self.audio_encoder = AudioEncoder(
            in_channels=self.model_config.spec_channels,
            out_channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.audio_encoder.kernel_size,
            dilation_rate=self.model_config.audio_encoder.dilation_rate,
            num_layers=self.model_config.audio_encoder.num_layers,
            cond_channels=self.embedded_speaker_channels,
        )
        self.flow = ResidualCouplingBlocks(
            channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.flow.kernel_size,
            dilation_rate=self.model_config.flow.dilation_rate,
            num_flows=self.model_config.flow.num_flows,
            num_layers=self.model_config.flow.num_layers_in_flow,
            cond_channels=self.embedded_speaker_channels,
            use_transformer_flow=self.model_config.flow.use_transformer_flow,
            use_SNAC=self.use_SNAC,
        )
        if self.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                in_channels=self.model_config.hidden_channels,
                hidden_channels=self.model_config.hidden_channels,
                kernel_size=self.model_config.duration_predictor.kernel_size,
                dropout_p=self.model_config.duration_predictor.dropout_p,
                num_flows=4,
                cond_channels=self.embedded_speaker_channels,
                language_emb_dim=self.embedded_language_dim,
            )
        else:
            self.duration_predictor = VitsDurationPredictor(
                in_channels=self.model_config.hidden_channels,
                hidden_channels=256,
                kernel_size=self.model_config.duration_predictor.kernel_size,
                dropout_p=self.model_config.duration_predictor.dropout_p,
                cond_channels=self.embedded_speaker_channels,
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
            cond_channels=self.embedded_speaker_channels,
            conv_pre_weight_norm=True,
            conv_post_weight_norm=True,
            conv_post_bias=False,
        )

    def monotonic_align(self, z_p, m_p, logs_p, x, x_mask, y_mask, mas_noise_scale=0.01):
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

            # vits2.0: add noice in Monotonic Align Search
            if mas_noise_scale > 0.0:
                epsilon = torch.std(logp) * torch.randn_like(logp) * mas_noise_scale
                logp = logp + epsilon

            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        return attn_durations, attn

    def get_speaker_embedding(self, speaker_embeds, speaker_ids, ref_spec):
        g = None  # [b, h, 1]
        if self.model_config.use_speaker_embeds and speaker_embeds is not None:
            g = F.normalize(speaker_embeds).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)
        elif self.model_config.use_speaker_ids and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)
        elif self.model_config.use_speaker_encoder and ref_spec is not None:
            g = self.speaker_encoder(ref_spec).unsqueeze(-1)  # [b, h, 1]
        return g

    def get_language_embedding(self, language_ids):
        lang_embed = None
        if self.model_config.use_language_ids:
            lang_embed = self.language_embedding(language_ids).unsqueeze(-1)  # [B, lang_channel, 1]
        return lang_embed

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
            ref_spec: [B, C, T_spec]: used in speaker encoder to generate the speaker embedding
            language_ids:`[B]: Batch of language ids.
        """
        g = self.get_speaker_embedding(speaker_embeds, speaker_ids, y)
        lang_embed = self.get_language_embedding(language_ids)  # [B, lang_channel, 1]

        # audio encoder, encode audio to embedding layer z's dimension
        z_q_audio, m_q_audio, logs_q_audio, y_mask = self.audio_encoder(
            x=y,
            x_lengths=y_lengths,
            g=g if not self.use_SNAC else None
        )

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(
            x=z_q_audio,
            x_lengths=y_lengths,
            segment_size=self.spec_segment_size,
            let_short_samples=True,
            pad_short=True
        )

        y_hat = self.waveform_decoder(
            x=z_slice,
            g=g if not self.use_SNAC else None
        )

        wav_seg = segment(
            x=waveform,
            segment_indices=slice_ids * self.config.audio.hop_length,
            segment_size=self.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        # flow layers
        z_q_dur, total_logdet = self.flow(
            x=z_q_audio,
            x_mask=y_mask,
            g=g
        )

        h_text, m_p_text, logs_p_text, x_mask = self.text_encoder(
            x=x,
            x_lengths=x_lengths,
            g=g if not self.use_SNAC else None,
            lang_emb=lang_embed
        )

        # monotonic align and duration predictor
        attn_durations, attn = self.monotonic_align(z_q_dur, m_p_text, logs_p_text, x, x_mask, y_mask, mas_noise_scale=self.mas_noise_scale)
        self.mas_noise_scale = max(self.mas_noise_scale - self.mas_noise_scale_decay, 0.0)

        # expand text token_size to audio token_size
        m_p_dur = torch.einsum("klmn, kjm -> kjn", [attn, m_p_text])
        logs_p_dur = torch.einsum("klmn, kjm -> kjn", [attn, logs_p_text])

        if self.use_sdp:
            loss_duration = self.duration_predictor(
                x=h_text.detach(),
                x_mask=x_mask,
                dr=attn_durations,
                g=g.detach() if g is not None else g,
                lang_emb=lang_embed.detach() if lang_embed is not None else lang_embed,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            predict_log_durations = self.duration_predictor(
                x=h_text.detach(),
                x_mask=x_mask,
                g=g.detach() if g is not None else g,
                lang_emb=lang_embed.detach() if lang_embed is not None else lang_embed,
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
            "m_p_dur": m_p_dur,  # [B, C, T_dec]
            "logs_p_dur": logs_p_dur,  # [B, C, T_dec]
            "z_q_dur": z_q_dur,  # [B, C, T_dec]
            "m_q_audio": m_q_audio,  # [B, C, T_dec]
            "logs_q_audio": logs_q_audio,  # [B, C, T_dec]
            "waveform_seg": wav_seg,  # [B, 1, spec_seg_size * hop_length]
            "slice_ids": slice_ids,
            "loss_duration": loss_duration,
            "total_logdet": total_logdet,
            "gt_speaker_emb": gt_speaker_emb,
            "syn_speaker_emb": syn_speaker_emb,
        }

    def forward_vae(
        self,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        speaker_embeds = None,
        speaker_ids = None,
    ) -> Dict:
        g = self.get_speaker_embedding(speaker_embeds, speaker_ids, y)

        # audio encoder, encode audio to embedding layer z's dimension
        z_q_audio, m_q_audio, logs_q_audio, y_mask = self.audio_encoder(
            x=y,
            x_lengths=y_lengths,
            g=g if not self.use_SNAC else None
        )

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(
            x=z_q_audio,
            x_lengths=y_lengths,
            segment_size=self.spec_segment_size,
            let_short_samples=True,
            pad_short=True
        )

        y_hat = self.waveform_decoder(
            x=z_slice,
            g=g if not self.use_SNAC else None
        )

        wav_seg = segment(
            x=waveform,
            segment_indices=slice_ids * self.config.audio.hop_length,
            segment_size=self.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        return {
            "y_hat": y_hat,  # [B, 1, T_wav]
            "m_q_audio": m_q_audio,  # [B, C, T_dec]
            "logs_q_audio": logs_q_audio,  # [B, C, T_dec]
            "waveform_seg": wav_seg,  # [B, 1, spec_seg_size * hop_length]
            "slice_ids": slice_ids,
        }

    def forward_SNAC_VC(
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
            ref_spec: [B, C, T_spec]: used in speaker encoder to generate the speaker embedding
            language_ids:`[B]: Batch of language ids.
        """
        g = self.get_speaker_embedding(speaker_embeds, speaker_ids, y)
        lang_embed = self.get_language_embedding(language_ids)  # [B, lang_channel, 1]

        # audio encoder, encode audio to embedding layer z's dimension
        z_q_audio, m_q_audio, logs_q_audio, y_mask = self.audio_encoder(
            x=y,
            x_lengths=y_lengths,
            g=g if not self.use_SNAC else None
        )

        # flow layers
        z_q_dur, fwd_logdet = self.flow(
            x=z_q_audio,
            x_mask=y_mask,
            g=g
        )

        h_text, m_p_text, logs_p_text, x_mask = self.text_encoder(
            x=x,
            x_lengths=x_lengths,
            g=g if not self.use_SNAC else None,
            lang_emb=lang_embed
        )

        # monotonic align and duration predictor
        attn_durations, attn = self.monotonic_align(z_q_dur, m_p_text, logs_p_text, x, x_mask, y_mask, mas_noise_scale=self.mas_noise_scale)
        self.mas_noise_scale = max(self.mas_noise_scale - self.mas_noise_scale_decay, 0.0)

        # expand text token_size to audio token_size
        m_p_dur = torch.einsum("klmn, kjm -> kjn", [attn, m_p_text])
        logs_p_dur = torch.einsum("klmn, kjm -> kjn", [attn, logs_p_text])
        z_p_dur = (m_p_dur + torch.randn_like(m_p_dur) * torch.exp(logs_p_dur)) * y_mask

        # flow backward layer
        z_p_audio, bwd_logdet = self.flow(
            x=z_p_dur,
            x_mask=y_mask,
            g=g,
            reverse=True
        )

        return {
            "logs_p_dur": logs_p_dur,  # [B, C, T_dec]
            "m_q_audio": m_q_audio,  # [B, C, T_dec]
            "logs_q_audio": logs_q_audio,  # [B, C, T_dec]
            "z_p_audio": z_p_audio,
            "total_logdet": bwd_logdet,
            "z_mask": y_mask
        }

    @torch.no_grad()
    def infer(
        self,
        x,  # [B, T_seq]
        x_lengths,  # [B]
        speaker_embeds=None,  # [T]
        speaker_ids=None,  # [B]
        ref_spec=None,  # [B, C, T_spec]
        language_ids=None,  # [B]
        noise_scale=1.0,
        length_scale=1.0,
        max_len=None
    ):
        g = self.get_speaker_embedding(speaker_embeds, speaker_ids, ref_spec)
        lang_embed = self.get_language_embedding(language_ids)  # [B, lang_channel, 1]

        h_text, m_p_text, logs_p_text, x_mask = self.text_encoder(
            x=x,
            x_lengths=x_lengths,
            g=g if not self.use_SNAC else None,
            lang_emb=lang_embed
        )

        if self.use_sdp:
            logw = self.duration_predictor(
                x=h_text,
                x_mask=x_mask,
                g=g,
                reverse=True,
                noise_scale=noise_scale,
                lang_emb=lang_embed
            )
        else:
            logw = self.duration_predictor(
                x=x,
                x_mask=x_mask,
                g=g,
                lang_emb=lang_embed
            )
        w = torch.exp(logw) * x_mask * length_scale

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p_dur = torch.matmul(attn.transpose(1, 2), m_p_text.transpose(1, 2)).transpose(1, 2)
        logs_p_dur = torch.matmul(attn.transpose(1, 2), logs_p_text.transpose(1, 2)).transpose(1, 2)

        z_p_dur = m_p_dur + torch.randn_like(m_p_dur) * torch.exp(logs_p_dur) * noise_scale
        z_p_audio, _ = self.flow(
            x=z_p_dur,
            x_mask=y_mask,
            g=g,
            reverse=True
        )

        z_p_audio = (z_p_audio * y_mask)[:, :, : max_len]
        wav = self.waveform_decoder(
            x=z_p_audio,
            g=g if not self.use_SNAC else None
        )
        return wav, attn, y_mask, (z_p_audio, z_p_dur, m_p_dur, logs_p_dur)

    @torch.no_grad()
    def generate_z_wav(self, spec, spec_len, speaker_id=None, speaker_embed=None):
        g = self.get_speaker_embedding(speaker_embed, speaker_id, spec)

        z_q_audio, _, _, _ = self.audio_encoder(
            x=spec,
            x_lengths=spec_len,
            g=g if not self.use_SNAC else None
        )
        wav = self.waveform_decoder(
            x=z_q_audio,
            g=g if not self.use_SNAC else None
        )
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

        z_q_audio, _, _, y_mask = self.audio_encoder(y, y_lengths, g=g_src)
        z_q_dur, _ = self.flow(z_q_audio, y_mask, g=g_src)
        z_p_audio, _ = self.flow(z_q_dur, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_p_audio * y_mask, g=g_tgt)
        return o_hat, y_mask, (z_q_audio, z_q_dur, z_p_audio)

    def voice_conversion_SNAC(self, y, y_lengths, ref_spec):
        """ test voice convert using SNAC mode in flow"""
        if not self.model_config.use_speaker_encoder:
            raise RuntimeError("voice_conversion_ref_wav only work in use_speaker_encoder mode")

        g_src = self.speaker_encoder(y).unsqueeze(-1)  # [b, h, 1]
        g_tgt = self.speaker_encoder(ref_spec).unsqueeze(-1)  # [b, h, 1]

        z_q_audio, _, _, y_mask = self.audio_encoder(y, y_lengths, g=None)
        z_q_dur, _ = self.flow(z_q_audio, y_mask, g=g_src)
        z_p_audio, _ = self.flow(z_q_dur, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_p_audio * y_mask, g=None)
        return o_hat, y_mask, (z_q_audio, z_q_dur, z_p_audio)