import math
import os
import time
import soundfile as sf
import torch
from coqpit import Coqpit
from typing import Dict, List, Union, Tuple
from util.gpu_mem_track import MemTracker, modelsize
from torch import nn
from torch.cuda.amp import autocast
from trainer import get_optimizer, get_scheduler
from torch.nn import functional as F
from config.config import NaturalTTSConfig
from language.languages import LanguageManager
from layers.discriminator import VitsDiscriminator
from layers.duration_predictor import VitsDurationPredictor
from layers.encoder import TextEncoder, AudioEncoder
from layers.flow import ResidualCouplingBlocks, AttentionFlow
from layers.generator import HifiganGenerator
from layers.learnable_upsampling import LearnableUpsampling
from layers.losses import NaturalSpeechDiscriminatorLoss, NaturalSpeechGeneratorLoss
from layers.quantizer import ResidualVectorQuantization
from layers.variance_predictor import DurationPredictor, PitchPredictor
from monotonic_align.maximum_path import maximum_path
from recipes.trainer_model import TrainerModelWithDataset
from text import text_to_tokens
from util.helper import segment, rand_segments
from util.mel_processing import wav_to_mel


class NaturalTTSModel(nn.Module):
    def __init__(self, config:NaturalTTSConfig, speaker_embed: torch.Tensor = None, language_manager: LanguageManager = None, share_durations = None ):
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.share_durations = share_durations
        self.speaker_embed = speaker_embed
        self.language_manager = language_manager

        # init multi-speaker, speaker_embedding is used when the speaker_embed is not provided
        self.num_speakers = self.model_config.num_speakers
        self.spec_segment_size = self.model_config.spec_segment_size
        self.embedded_speaker_dim = self.model_config.speaker_embedding_channels
        if self.num_speakers > 0:
            self.speaker_embedding = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)
        self.use_gt_duration = self.model_config.use_gt_duration

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
        self.flow = AttentionFlow(
            channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.flow.kernel_size,
            dilation_rate=self.model_config.flow.dilation_rate,
            num_layers=self.model_config.flow.num_layers_in_flow,
            num_flows=self.model_config.flow.num_flows,
            cond_channels=self.embedded_speaker_dim,
            attention_heads=self.model_config.flow.attention_heads
        )
        self.duration_predictor = DurationPredictor(
            channels=self.model_config.hidden_channels,
            condition_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.duration_predictor.kernel_size,
            n_stack=self.model_config.duration_predictor.n_stack,
            n_stack_in_stack=self.model_config.duration_predictor.n_stack_in_stack,
            attention_num_head=self.model_config.duration_predictor.attention_num_head,
            dropout_p=self.model_config.duration_predictor.dropout_p
        )
        self.pitch_predictor = PitchPredictor(
            channels=self.model_config.hidden_channels,
            condition_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.pitch_predictor.kernel_size,
            n_stack=self.model_config.pitch_predictor.n_stack,
            n_stack_in_stack=self.model_config.pitch_predictor.n_stack_in_stack,
            attention_num_head=self.model_config.pitch_predictor.attention_num_head,
            dropout_p=self.model_config.pitch_predictor.dropout_p
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
            cond_channels=0,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False
        )
        self.quantizer = ResidualVectorQuantization(
            num_quantizers=self.model_config.quantizer.num_quantizers,
            codebook_size=self.model_config.quantizer.codebook_size,
            codebook_dim=self.model_config.quantizer.codebook_dimension
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

    def match_mel_token(self, z_p, m_p, logs_p, x_mask, y_mask):
        """
        match the mel and token to get the duration. duration in the source dataset is not used because it may be wrong.
        """
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            # code here does: using "Probability density function", by applying z_p to Probability density function
            # to get the probability
            o_scale = torch.exp(-2 * logs_p)  # 1/p**2, p is the Variance
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1] log( 1/sqrt(2*pi*p**2) )
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p ** 2)])  # log( -x**2/(2*p**2) ), x will be z_p here
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])  # log( 2xu/2p**2 ), u is the mean here
            logp4 = torch.sum(-0.5 * (m_p ** 2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]  log( u**2/(2*p**2) )
            logp = logp2 + logp3 + logp1 + logp4  # log("Probability density")
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        durations = attn.sum(3)
        return durations, attn

    def forward(
        self,
        x,  # [B,C,T]
        x_lengths,  # [B]
        y,  # [B,specC,specT]
        y_lengths,  # [B]
        pitch=None,
        duration=None,
        filenames=None,
        speaker_prompt=None,
    ):
        # audio encoder, encode audio to embedding layer z's dimension,
        # z:[B,C,spectrogramToken] m_q:[B,C,specT] logs_q:[B,C,specT] z_mask:[B,1,specT]
        z, m_q, logs_q, z_mask = self.audio_encoder(
            x=y,
            x_lengths=y_lengths,
            g=None
        )

        # # generate prompt from x, use half x_length
        # prompt_length = y_lengths.min() // 2
        # prompts, ids_prompts = rand_segments(
        #     x=z,
        #     x_lengths=y_lengths,
        #     segment_size=prompt_length,
        #     let_short_samples=True,
        #     pad_short=True
        # )
        #
        # # z_p:[B,C,specT]
        # z_p = self.flow(
        #     x=z,
        #     x_mask=z_mask,
        #     g=None,
        #     condition=prompts
        # )

        # quantize z using RVQ
        z_quant, _, _ = self.quantizer(z)

        z_slice, ids_slice = rand_segments(
            x=z_quant,
            x_lengths=y_lengths,
            segment_size=self.spec_segment_size,
            let_short_samples=True,
            pad_short=True
        )
        y_hat = self.waveform_decoder(z_slice, g=None)

        # x, m_p, logs_p, x_mask = self.text_encoder(
        #     x=x,
        #     x_lengths=x_lengths,
        #     lang_emb=None
        # )

        # # if duration is None:
        # #     duration_gt, attn = self.match_mel_token(z_p, m_p, logs_p, x_mask, z_mask)
        # #     for i in range(len(filenames)):
        # #         self.share_durations.set(filenames[i], duration_gt[i][0][0:x_lengths[i]])
        # # else:
        # #     duration_gt = duration
        #
        # duration_gt, attn = self.match_mel_token(z_p, m_p, logs_p, x_mask, z_mask)
        #
        # # # predict durator
        # # duration_logw = self.duration_predictor(
        # #     x=x,
        # #     masks=x_mask,
        # #     speech_prompts=prompts
        # # )  # duration_logw:[B,1,T]
        # # duration_p = torch.exp(duration_logw.unsqueeze(1)) * x_mask
        # # duration_logw_gt = torch.log(duration_gt.unsqueeze(1) + 1e-6) * x_mask
        # # duration_loss = torch.sum((duration_logw - duration_logw_gt) ** 2, [1, 2]) / torch.sum(x_mask)
        # #
        # # pitch_logw, pitch_embed = self.pitch_predictor(
        # #     x=x,
        # #     masks=x_mask,
        # #     speech_prompts=prompts
        # # )  # pitch_logw:[B,1,T]
        # # x = x + pitch_embed
        # #
        # # pitch_p = torch.exp(pitch_logw.unsqueeze(1)) * x_mask
        # # # map ground true pitch to the same dimention of x
        # # pitch = torch.einsum("bcxy,bcy->bcx", attn, pitch.unsqueeze(1))
        # # pitch = (pitch / (attn.sum(3) + 1e-6) * x_mask).round(decimals=3)
        # # # compute pitch loss
        # # pitch_logw_gt = torch.log(pitch.unsqueeze(1) + 1e-6) * x_mask
        # # pitch_loss = torch.sum((pitch_logw - pitch_logw_gt) ** 2, [1, 2]) / torch.sum(x_mask)

        # duration_loss = torch.zeros_like(x_mask)
        # pitch_loss = torch.zeros_like(x_mask)
        #
        # # differentiable durator (learnable upsampling)
        # upsampled_rep, p_mask, _, W, C = self.learnable_upsampling(
        #     # duration=duration_gt.squeeze() if self.use_gt_duration else duration_p.squeeze(),
        #     duration=duration_gt.squeeze(),
        #     tokens=x.transpose(1, 2),
        #     src_len=x_lengths,
        #     src_mask=~(x_mask.squeeze(1).bool()),
        #     max_src_len=x_lengths.max(),
        # )
        # p_mask = p_mask.unsqueeze(1)
        # m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)
        # x_p = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)) * p_mask
        #
        # z_q = self.flow(
        #     x=x_p,
        #     x_mask=p_mask,
        #     g=None,
        #     condition=prompts,
        #     reverse=True
        # )
        #
        # z_q_quant, _, _ = self.quantizer(z_q)

        # z_q_lengths = p_mask.flatten(1, -1).sum(dim=-1).long()
        # z_slice_q, ids_slice_q = rand_segments(
        #     x=z_q_quant,
        #     x_lengths=torch.minimum(z_q_lengths, y_lengths),
        #     segment_size=self.spec_segment_size,
        #     let_short_samples=True,
        #     pad_short=True
        # )
        #
        # y_hat_e2e = self.waveform_decoder(z_slice_q, g=None)
        #
        # return {
        #     "y_hat": y_hat,  # the generated waveform
        #     "duration": duration_gt,
        #     "duration_loss": duration_loss,
        #     "pitch": pitch,
        #     "pitch_loss": pitch_loss,
        #     "ids_slice": ids_slice,
        #     "x_mask": x_mask,
        #     "z_mask": z_mask,
        #     "z": z,
        #     "z_p": z_p,
        #     "m_p": m_p,
        #     "logs_p": logs_p,
        #     "m_q": m_q,
        #     "logs_q": logs_q,
        #     "p_mask": p_mask,
        #     "W": W,
        #     "y_hat_e2e": y_hat_e2e,
        #     "z_q": z_q,
        #     "ids_slice_q": ids_slice_q
        # }

        return {
            "y_hat": y_hat,  # the generated waveform
            "duration": torch.zeros([1]),
            "duration_loss": torch.zeros([1]),
            "pitch": pitch,
            "pitch_loss": torch.zeros([1]),
            "ids_slice": ids_slice,
            "x_mask": torch.zeros([1]),
            "z_mask": z_mask,
            "z": z,
            "z_p": torch.zeros([1]),
            "m_p": torch.zeros([1]),
            "logs_p": torch.zeros([1]),
            "m_q": m_q,
            "logs_q": logs_q,
            "p_mask": torch.zeros([1]),
            "W": torch.zeros([1]),
            "y_hat_e2e": torch.zeros([1]),
            "z_q": torch.zeros([1]),
            "ids_slice_q": torch.zeros([1])
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

        logw_p = self.duration_predictor(x, x_mask, g=g)
        w = torch.exp(logw_p) * x_mask * self.model_config.length_scale
        if duration is not None:
            w = duration.unsqueeze(1) * x_mask * self.model_config.length_scale

        upsampled_rep, p_mask, _, W, C = self.learnable_upsampling(
            duration=w.squeeze(1),
            V=x.transpose(1, 2),
            src_len=x_lengths,
            src_mask=~(x_mask.squeeze(1).bool()),
            tgt_len=None,
            max_src_len=x_mask.shape[-1]
        )
        m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)

        y_mask = p_mask.unsqueeze(1)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.model_config.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # quantize with RVQ
        z = self.quantizer.embed(z)

        z = (z * y_mask)[:, :, :self.model_config.max_inference_len]
        y_hat = self.waveform_decoder(z, g=g)
        return y_hat

    @torch.no_grad()
    def generate_wav(self, z):
        z = z[0].unsqueeze(0)
        z_quant, _, _ = self.quantizer(z)
        wav = self.waveform_decoder(z_quant, g=None)
        return wav


class NaturalTTSTrain(TrainerModelWithDataset):
    """
    Natural Speech model training model.
    """
    def __init__(self, config:NaturalTTSConfig, speaker_embed: torch.Tensor = None, language_manager: LanguageManager = None, share_vars = None):
        super().__init__(share_vars)
        self.config = config
        self.model_config = config.model

        self.generator = NaturalTTSModel(
            config=config,
            speaker_embed=speaker_embed,
            language_manager=language_manager,
            share_durations = self.share_vars
        )

        self.discriminator = VitsDiscriminator(
            periods=self.model_config.discriminator.periods_multi_period,
            use_spectral_norm=self.model_config.discriminator.use_spectral_norm,
        )

    def train_step(self, batch: Dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        spec_lens = batch["spec_lens"]
        if optimizer_idx == 0:
            # print(batch["raw_texts"])

            tokens = batch["tokens"]
            token_lens = batch["token_lens"]
            spec = batch["spec"]
            speaker_embeds = batch["speaker_embed"]
            wav = batch["waveform"]
            pitch = batch["pitch"]
            duration = batch["duration"]
            filenames = batch["filenames"]

            # generator pass
            outputs = self.generator(
                x=tokens,
                x_lengths=token_lens,
                y=spec,
                y_lengths=spec_lens,
                pitch=pitch,
                duration=duration,
                filenames=filenames
            )

            wav_slice = segment(
                x=wav,
                segment_indices=outputs["ids_slice"] * self.config.audio.hop_length,
                segment_size=self.model_config.spec_segment_size * self.config.audio.hop_length,
                pad_short=True
            )
            # wav_slice_q = segment(
            #     x=wav,
            #     segment_indices=outputs["ids_slice_q"] * self.config.audio.hop_length,
            #     segment_size=self.model_config.spec_segment_size * self.config.audio.hop_length,
            #     pad_short=True
            # )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs
            self.wav_slice = wav_slice
            # self.wav_slice_q = wav_slice_q

            # compute scores and features
            scores_disc_real, _, scores_disc_fake, _ = self.discriminator(
                x=wav_slice,
                x_hat=outputs["y_hat"].detach()
            )
            # scores_disc_real_e2e, _, scores_disc_fake_e2e, _ = self.discriminator(
            #     x=wav_slice_q,
            #     x_hat=outputs["y_hat_e2e"].detach()
            # )

            # compute loss
            with autocast(enabled=False):
                loss_dict = criterion[0](
                    scores_disc_real=scores_disc_real,
                    scores_disc_fake=scores_disc_fake,
                )
                # loss_dict_e2e = criterion[0](
                #     scores_disc_real=scores_disc_real_e2e,
                #     scores_disc_fake=scores_disc_fake_e2e,
                # )
                # # add loss and e2e_loss, but the dicrimator loss use the value in lost_dict
                # loss_dict["loss"] = loss_dict["loss"] + loss_dict_e2e["loss"]

            return outputs, loss_dict

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
                x=self.wav_slice,
                x_hat=self.model_outputs_cache["y_hat"]
            )
            # scores_disc_real_e2e, _, scores_disc_fake_e2e, _ = self.discriminator(
            #     x=self.wav_slice_q,
            #     x_hat=self.model_outputs_cache["y_hat_e2e"]
            # )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                # loss_dict = criterion[1](
                #     mel_slice=mel_slice,
                #     mel_slice_hat=mel_slice_hat,
                #     scores_disc_fake=scores_disc_fake,
                #     scores_disc_fake_e2e=scores_disc_fake_e2e,
                #     feats_disc_real=feats_disc_real,
                #     feats_disc_fake=feats_disc_fake,
                #     duration_loss=self.model_outputs_cache["duration_loss"],
                #     pitch_loss = self.model_outputs_cache["pitch_loss"],
                #     z_p=self.model_outputs_cache["z_p"],
                #     m_p=self.model_outputs_cache["m_p"],
                #     logs_p=self.model_outputs_cache["logs_p"],
                #     z_q=self.model_outputs_cache["z_q"],
                #     m_q=self.model_outputs_cache["m_q"],
                #     logs_q=self.model_outputs_cache["logs_q"],
                #     p_mask=self.model_outputs_cache["p_mask"],
                #     z_mask=self.model_outputs_cache["z_mask"],
                # )

                loss_dict = criterion[1](
                    mel_slice=mel_slice,
                    mel_slice_hat=mel_slice_hat,
                    scores_disc_fake=scores_disc_fake,
                    scores_disc_fake_e2e=torch.zeros([0]),
                    feats_disc_real=feats_disc_real,
                    feats_disc_fake=feats_disc_fake,
                    duration_loss=self.model_outputs_cache["duration_loss"],
                    pitch_loss = self.model_outputs_cache["pitch_loss"],
                    z_p=self.model_outputs_cache["z_p"],
                    m_p=self.model_outputs_cache["m_p"],
                    logs_p=self.model_outputs_cache["logs_p"],
                    z_q=self.model_outputs_cache["z_q"],
                    m_q=self.model_outputs_cache["m_q"],
                    logs_q=self.model_outputs_cache["logs_q"],
                    p_mask=self.model_outputs_cache["p_mask"],
                    z_mask=self.model_outputs_cache["z_mask"],
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        output, lass_dict = self.train_step(batch, criterion, optimizer_idx)
        if optimizer_idx == 1:
            wav = self.generator.generate_wav(output["z"])
            wav = wav[0, 0].cpu().float().numpy()
            filename = os.path.basename(batch["filenames"][0])
            sf.write(f"{self.config.output_path}/{filename}_{int(time.time())}.wav", wav, 22050)

        return output, lass_dict

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

    def test_run(self, assets: Dict):
        pass