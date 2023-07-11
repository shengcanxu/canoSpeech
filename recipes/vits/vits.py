import math
import platform
from itertools import chain
from typing import Dict, List, Union, Tuple
from trainer.torch import DistributedSampler
import torchaudio
from torch.cuda.amp.autocast_mode import autocast
import torch
from torch.nn import functional as F
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from trainer import TrainerModel
from trainer.trainer_utils import get_optimizer, get_scheduler
from config.config import VitsConfig
from dataset.dataset import TextAudioDataset
from dataset.sampler import DistributedBucketSampler
from language.languages import LanguageManager
from layers.discriminator import VitsDiscriminator
from layers.duration_predictor import DurationPredictor
from layers.flow import ResidualCouplingBlocks
from layers.generator import HifiganGenerator
from layers.losses import VitsDiscriminatorLoss, VitsGeneratorLoss
from layers.encoder import TextEncoder, AudioEncoder
from monotonic_align.maximum_path import maximum_path
from speaker.speakers import SpeakerManager
from util.helper import sequence_mask, segment, rand_segments
from util.mel_processing import wav_to_spec, spec_to_mel, wav_to_mel


class VitsModel(nn.Module):
    def __init__(self, config:VitsConfig, speaker_manager: SpeakerManager = None, language_manager: LanguageManager = None, ):
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager

        self.init_multispeaker(config)
        self.init_multilingual(config)
        self.init_upsampling()

        self.spec_segment_size = self.model_config.spec_segment_size

        self.text_encoder = TextEncoder(
            n_vocab=self.model_config.text_encoder.num_chars,
            out_channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            hidden_channels_ffn=self.model_config.text_encoder.hidden_channels_ffn_text_encoder,
            num_heads=self.model_config.text_encoder.num_heads_text_encoder,
            num_layers=self.model_config.text_encoder.num_layers_text_encoder,
            kernel_size=self.model_config.text_encoder.kernel_size_text_encoder,
            dropout_p=self.model_config.text_encoder.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )

        self.audio_encoder = AudioEncoder(
            in_channels=self.model_config.out_channels,
            out_channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.audio_encoder.kernel_size_audio_encoder,
            dilation_rate=self.model_config.audio_encoder.dilation_rate_audio_encoder,
            num_layers=self.model_config.audio_encoder.num_layers_audio_encoder,
            cond_channels=self.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            channels=self.model_config.hidden_channels,
            hidden_channels=self.model_config.hidden_channels,
            kernel_size=self.model_config.flow.kernel_size_flow,
            dilation_rate=self.model_config.flow.dilation_rate_flow,
            num_layers=self.model_config.flow.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        self.duration_predictor = DurationPredictor(
            in_channels=self.model_config.hidden_channels,
            hidden_channels=256,
            kernel_size=3,
            dropout_p=self.model_config.duration_predictor.dropout_p_duration_predictor,
            cond_channels=self.embedded_speaker_dim,
            language_emb_dim=self.embedded_language_dim,
        )

        self.waveform_decoder = HifiganGenerator(
            in_channels=self.model_config.hidden_channels,
            out_channels=1,
            resblock_type=self.model_config.waveform_decoder.resblock_type_decoder,
            resblock_dilation_sizes=self.model_config.waveform_decoder.resblock_dilation_sizes_decoder,
            resblock_kernel_sizes=self.model_config.waveform_decoder.resblock_kernel_sizes_decoder,
            upsample_kernel_sizes=self.model_config.waveform_decoder.upsample_kernel_sizes_decoder,
            upsample_initial_channel=self.model_config.waveform_decoder.upsample_initial_channel_decoder,
            upsample_factors=self.model_config.waveform_decoder.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    def _init_speaker_embedding(self):
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.model_config.speaker_embedding_channels
            self.speaker_embedding = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.
        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.
        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.model_config.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.model_config.use_speaker_embedding:
            self._init_speaker_embedding()

        # if self.model_config.use_d_vector_file:
        #     self._init_d_vector()

        # if self.model_config.use_speaker_encoder_as_loss:
            # if self.speaker_manager.encoder is None and (
            #     not self.model_config.speaker_encoder_model_path or not self.model_config.speaker_encoder_config_path
            # ):
            #     raise RuntimeError(
            #         " [!] To use the speaker consistency loss (SCL) you need to specify speaker_encoder_model_path and speaker_encoder_config_path !!"
            #     )
            #
            # self.speaker_manager.encoder.eval()
            # print(" > External Speaker Encoder Loaded !!")
            #
            # if (
            #     hasattr(self.speaker_manager.encoder, "audio_config")
            #     and self.config.audio.sample_rate != self.speaker_manager.encoder.audio_config["sample_rate"]
            # ):
            #     self.audio_transform = torchaudio.transforms.Resample(
            #         orig_freq=self.config.audio.sample_rate,
            #         new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
            #     )

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

    def init_upsampling(self):
        """ Initialize upsampling modules of a model. """
        # if self.model_config.encoder_sample_rate:
        #     self.interpolate_factor = self.config.audio["sample_rate"] / self.model_config.encoder_sample_rate
        #     self.audio_resampler = torchaudio.transforms.Resample(
        #         orig_freq=self.config.audio["sample_rate"], new_freq=self.model_config.encoder_sample_rate
        #     )
        pass

    def forward(
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        d_vectors = None,
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
            aux_input (dict, optional): Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.
                d_vectors:[B, C, 1]` speaker_ids:`[B]`  language_ids:`[B]`
        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
        """
        outputs = {}
        sid, g, lid = self._set_cond_input(d_vectors, speaker_ids, language_ids)
        # speaker embedding
        if self.model_config.use_speaker_embedding and sid is not None:
            g = self.speaker_embedding(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.model_config.use_language_embedding and lid is not None:
            lang_emb = self.language_embedding(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)

        # posterior encoder
        z, m_q, logs_q, y_mask = self.audio_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # duration predictor
        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb)

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        if self.model_config.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None

        outputs.update({
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "m_p": m_p,
            "logs_p": logs_p,
            "z": z,
            "z_p": z_p,
            "m_q": m_q,
            "logs_q": logs_q,
            "waveform_seg": wav_seg,
            "gt_spk_emb": gt_spk_emb,
            "syn_spk_emb": syn_spk_emb,
            "slice_ids": slice_ids,
        })
        return outputs

    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
        log_durations = self.duration_predictor(
            x.detach(),
            x_mask,
            g=g.detach() if g is not None else g,
            lang_emb=lang_emb.detach() if lang_emb is not None else lang_emb,
        )
        loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn

    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.spec_segment_size
        # if self.model_config.encoder_sample_rate:
        #     # recompute the slices and spec_segment_size if needed
        #     slice_ids = slice_ids * int(self.interpolate_factor) if slice_ids is not None else slice_ids
        #     spec_segment_size = spec_segment_size * int(self.interpolate_factor)
        #     # interpolate z if needed
        #     if self.model_config.interpolate_z:
        #         z = torch.nn.functional.interpolate(z, scale_factor=[self.interpolate_factor], mode="linear").squeeze(0)
        #         # recompute the mask if needed
        #         if y_lengths is not None and y_mask is not None:
        #             y_mask = (
        #                 sequence_mask(y_lengths * self.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
        #             )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask

    @staticmethod
    def _set_cond_input(d_vectors, speaker_ids, language_ids):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid, durations = None, None, None, None
        if speaker_ids is not None:
            sid = speaker_ids
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)
        if d_vectors is not None:
            g = F.normalize(d_vectors).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if language_ids is not None:
            lid = language_ids
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)
        return sid, g, lid


class VitsTrain(TrainerModel):
    def __init__(self, config:VitsConfig, speaker_manager: SpeakerManager = None, language_manager: LanguageManager = None, ):
        super().__init__()
        self.config = config
        self.model_config = config.model

        self.generator = VitsModel(
            config=config,
            speaker_manager=speaker_manager,
            language_manager=language_manager
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

        dataset = TextAudioDataset(samples, config)

        # wait all the DDP process to be ready
        if num_gpus > 1:
            dist.barrier()

        sampler = DistributedSampler(dataset) if num_gpus > 1 else RandomSampler(dataset)

        # set num_workers>0 the DataLoader will be very slow in windows, because it re-start
        # all processes every epoch. https://github.com/JaidedAI/EasyOCR/issues/274
        num_workers = config.dataset_config.num_eval_loader_workers if is_eval else config.dataset_config.num_loader_workers
        if platform.system() == "Windows":
            num_workers = 0
        loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.eval_batch_size if is_eval else config.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            # persistent_workers=True,
        )
        return loader

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device. datas are send to GPU before calling this func"""
        wav = batch["waveform"]
        audio_config = self.config.audio
        spec = wav_to_spec(
            wav=wav,
            n_fft=audio_config.fft_length,
            hop_size=audio_config.hop_length,
            win_size=audio_config.win_length,
            center=False
        )
        batch["spec"] = spec

        mel = spec_to_mel(
            spec=spec,
            n_fft=audio_config.fft_length,
            num_mels=audio_config.num_mels,
            sample_rate=audio_config.sample_rate,
            fmin=audio_config.mel_fmin,
            fmax=audio_config.mel_fmax
        )
        batch["mel"] = mel

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

    def train_step(self, batch: Dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        spec_lens = batch["spec_lens"]
        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lens = batch["token_lens"]
            spec = batch["spec"]
            d_vectors = None
            speaker_ids = None
            language_ids = None
            waveform = batch["waveform"]

            # generator pass
            outputs = self.generator(
                x=tokens,
                x_lengths=token_lens,
                y=spec,
                y_lengths=spec_lens,
                waveform=waveform,
                d_vectors=None,
                speaker_ids=None,
                language_ids=None
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs

            # compute scores and features
            scores_disc_fake, _, scores_disc_real, _ = self.discriminator(
                x=outputs["model_outputs"].detach(),
                x_hat=outputs["waveform_seg"]
            )

            # compute loss
            with autocast(enabled=False):
                loss_dict = criterion[0](
                    scores_disc_real=scores_disc_real,
                    scores_disc_fake=scores_disc_fake,
                )
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
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_length,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.discriminator(
                self.model_outputs_cache["model_outputs"],
                self.model_outputs_cache["waveform_seg"]
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
                    gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                    syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in train_step()`"""
        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters."""
        # select generator parameters
        optimizer0 = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr,
            model=self.discriminator
        )

        # gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("discriminator."))
        optimizer1 = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr,
            model=self.generator
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

    def forward(self, input: torch.Tensor, *args, aux_input={}, **kwargs) -> Dict:
        print("nothing to do! doing the real train code in train_step. ")
        return input
