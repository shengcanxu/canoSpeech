import math
import random
from typing import Dict, List, Union, Tuple

from dataset.basic_dataset import TextAudioDataset
from dataset.sampler import DistributedBucketSampler
from torch.cuda.amp.autocast_mode import autocast
import torch
import os
import time
from torch.nn import functional as F
from coqpit import Coqpit
from torch import nn
from trainer.trainer_utils import get_optimizer, get_scheduler
from config.config import VitsConfig
from language.languages import LanguageManager
from layers.discriminator import VitsDiscriminator
from layers.duration_predictor import VitsDurationPredictor, generate_path
from layers.flow import ResidualCouplingBlocks
from layers.generator import HifiganGenerator
from layers.losses import VitsDiscriminatorLoss, VitsGeneratorLoss
from layers.encoder import TextEncoder, AudioEncoder
from monotonic_align.maximum_path import maximum_path
from recipes.trainer_model import TrainerModelWithDataset
from text import text_to_tokens, _intersperse
from util.helper import sequence_mask, segment, rand_segments
from util.mel_processing import wav_to_spec, spec_to_mel, wav_to_mel
import soundfile as sf


class VitsModel(nn.Module):
    def __init__(self, config:VitsConfig, speaker_embed: torch.Tensor = None, language_manager: LanguageManager = None ):
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
        g = None
        if speaker_embeds is not None:
            g = F.normalize(speaker_embeds).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        # speaker embedding
        if self.model_config.use_speaker_ids and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)  # [b, h, 1]

        # audio encoder, encode audio to embedding layer z's dimension
        z, m_q, logs_q, y_mask = self.audio_encoder(y, y_lengths, g=g)

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        y_hat = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            self.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        # # for test
        # m_p = logs_p = z_p = loss_duration = torch.LongTensor([0])

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=None)

        # monotonic align and duration predictor
        attn_durations, attn = self.monotonic_align(z_p, m_p, logs_p, x, x_mask, y_mask)
        # expand text token_size to audio token_size
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
        log_durations = self.duration_predictor(
            x.detach(),
            x_mask,
            g=g.detach() if g is not None else g,
            lang_emb=None,
        )
        loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)

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
        }

    @torch.no_grad()
    def infer(
        self,
        x,  # [B, T_seq]
        x_lengths,  # [B]
        speaker_ids=None,  # [B]
        noise_scale=1.0,
        length_scale=1.0,
        max_len=None
    ):
        # speaker embedding
        if self.num_speakers > 1 and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=None)

        logw = self.duration_predictor(
            x=x,
            x_mask=x_mask,
            g=g if g is not None else g,
            lang_emb=None
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
    def generate_z_wav(self, spec, spec_len, speaker_id):
        g = self.speaker_embedding(speaker_id).unsqueeze(-1)  # [b, h, 1]
        z, _, _, _ = self.audio_encoder(spec, spec_len, g=g)
        wav = self.waveform_decoder(z, g=g)
        return wav

class VitsTrain(TrainerModelWithDataset):
    """
    VITS and YourTTS model training model.
    """
    def __init__(self, config:VitsConfig, speaker_embed: torch.Tensor = None, language_manager: LanguageManager = None, ):
        super().__init__(config)
        self.config = config
        self.model_config = config.model
        self.balance_disc_generator = config.balance_disc_generator
        self.skip_discriminator = False

        self.generator = VitsModel(
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
            waveform = batch["waveform"]
            speaker_ids = batch["speaker_ids"]

            # generator pass
            outputs = self.generator(
                x=tokens,
                x_lengths=token_lens,
                y=spec,
                y_lengths=spec_lens,
                waveform=waveform,
                speaker_embeds=None,
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

                if self.balance_disc_generator:
                    loss_dict["loss_disc"] = self.disc_loss_dict["loss_disc"]
                    loss_dict["loss_disc_real_all"] = self.disc_loss_dict["loss_disc_real_all"]
                    loss_dict["loss_disc_fake_all"] = self.disc_loss_dict["loss_disc_fake_all"]
                    # auto balance discriminator and generator, make sure loss of disciminator will be roughly 1.5x - 2.0x of generator
                    self.skip_discriminator = loss_dict["loss_disc"] < 0.4 * self.config.loss.disc_loss_alpha

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

    def get_sampler(self, config: Coqpit, dataset, num_gpus=1, rank=0):
        return DistributedBucketSampler(
            dataset=dataset,
            batch_size=config.batch_size,
            boundaries=[32,300,400,500,600,700,800,900,1000],
            num_replicas=num_gpus,
            rank=0,
            shuffle=True
        )

    def get_dataset(self, config: Coqpit, samples):
        return TextAudioDataset(samples, config)

    def forward(self, input: torch.Tensor) -> Dict:
        print("nothing to do! doing the real train code in train_step. ")
        return input

    def inference(self, text:str, speaker_id:int=None):
        tokens = text_to_tokens(text)
        if self.config.text.add_blank:
            tokens = _intersperse(tokens, 0)
        tokens = torch.LongTensor(tokens).unsqueeze(dim=0).cuda()
        x_lengths = torch.LongTensor([tokens.size(1)]).cuda()
        speaker_ids = torch.LongTensor([speaker_id]).cuda() if speaker_id is not None else None

        self.generator.eval()
        wav, _, _, _ = self.generator.infer(
            tokens,
            x_lengths,
            speaker_ids=speaker_ids,
            noise_scale=0.667,
            length_scale=1,
        )
        return wav

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        output, loss_dict = self.train_step(batch, criterion, optimizer_idx)
        if optimizer_idx == 0:
            spec = batch["spec"][[0]]
            spec_len = batch["spec_lens"][[0]]
            speaker_id = batch["speaker_ids"][[0]]
            wav = self.generator.generate_z_wav(spec, spec_len, speaker_id)

            wav = wav[0, 0].cpu().float().numpy()
            filename = os.path.basename(batch["filenames"][0])
            sf.write(f"{self.config.output_path}/{int(time.time())}_{filename}", wav, 22050)

        return output, loss_dict

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        output_path = assets["output_path"]

        print("doing test run...")
        text = "Who else do you want to talk to? You can go with me today to the meeting."
        speaker_id = random.randint(0, 9)
        wav = self.inference(text, speaker_id)
        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/test_{int(time.time())}.wav", wav, 22050)
