from dataclasses import dataclass, field, asdict
from typing import List

from coqpit import Coqpit, check_argument
from trainer import TrainerConfig


@dataclass
class TTSDatasetConfig(Coqpit):
    """Base config for TTS datasets.
    Args:
        formatter (str):
            Formatter name that defines used formatter in ```TTS.tts.datasets.formatter```. Defaults to `""`.
        dataset_name (str):
            Unique name for the dataset. Defaults to `""`.
        path (str):
            Root path to the dataset files. Defaults to `""`.
        meta_file_train (str):
            Name of the dataset meta file. Or a list of speakers to be ignored at training for multi-speaker datasets.
            Defaults to `""`.
        ignored_speakers (List):
            List of speakers IDs that are not used at the training. Default None.
        language (str):
            Language code of the dataset. If defined, it overrides `phoneme_language`. Defaults to `""`.
        meta_file_val (str):
            Name of the dataset meta file that defines the instances used at validation.
        meta_file_attn_mask (str):
            Path to the file that lists the attention mask files used with models that require attention masks to
            train the duration predictor.
    """

    dataset_name: str = ""
    path: str = ""
    use_cache:bool = False
    meta_file_train: str = ""
    meta_file_val: str = ""
    ignored_speakers: List[str] = None
    language: str = "en"

    num_loader_workers:int = 8
    num_eval_loader_workers:int = 8
    melspec_use_GPU:bool = False

@dataclass
class TextConfig(Coqpit):
    min_text_len:int = 1
    max_text_len:int = 190
    add_blank:bool = True
    cleaned_text:bool = True
    text_cleaners:List[str] = None

@dataclass
class AudioConfig(Coqpit):
    mel_fmin: float = 0
    mel_fmax = None
    hop_length:int = 256
    win_length:int = 1024
    sample_rate:int = 16000
    fft_length:int = 1024
    num_mels:int = 80

@dataclass
class TextEncoderConfig(Coqpit):
    num_chars:int = 165
    hidden_channels_ffn_text_encoder:int = 768
    num_heads_text_encoder:int = 2
    num_layers_text_encoder:int = 10
    kernel_size_text_encoder:int = 3
    dropout_p_text_encoder:float = 0.1

@dataclass
class AudioEncoderConfig(Coqpit):
    kernel_size_audio_encoder:int = 5
    dilation_rate_audio_encoder:int = 1
    num_layers_audio_encoder:int = 16

@dataclass
class FlowConfig(Coqpit):
    kernel_size_flow:int = 5
    dilation_rate_flow:int = 1
    num_layers_flow:int = 4

@dataclass
class DurationPredictorConfig(Coqpit):
    dropout_p_duration_predictor:float = 0.5

@dataclass
class WaveformDecoderConfig(Coqpit):
    resblock_type_decoder:str = "2"
    resblock_dilation_sizes_decoder:List[List[int]] = field(default_factory=lambda: [
                [ 1, 3, 5 ],
                [ 1, 3, 5 ],
                [ 1, 3, 5 ]
            ])
    resblock_kernel_sizes_decoder:List[int] = field(default_factory=lambda: [ 3, 7, 11 ])
    upsample_kernel_sizes_decoder:List[int] = field(default_factory=lambda: [ 16, 16, 4, 4 ])
    upsample_initial_channel_decoder:int = 512
    upsample_rates_decoder:List[int] = field(default_factory=lambda: [ 8, 8, 2, 2 ])

@dataclass
class DiscriminatorConfig(Coqpit):
    periods_multi_period_discriminator:List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_spectral_norm_disriminator:bool = False

@dataclass
class VitsModelConfig(Coqpit):
    hidden_channels: int = 192
    out_channels:int = 513
    spec_segment_size:int = 32

    language_embedding_channels: int = 4
    use_language_embedding:bool = False
    language_ids_file:str = None
    num_speakers:int = 0
    speaker_embedding_channels:int = 256
    use_speaker_embedding:bool = False
    use_speaker_encoder_as_loss:bool = False

    text_encoder: TextEncoderConfig = field(default_factory=lambda: TextEncoderConfig())
    audio_encoder: AudioEncoderConfig = field(default_factory=lambda: AudioEncoderConfig())
    flow:FlowConfig = field(default_factory=lambda:FlowConfig())
    duration_predictor:DurationPredictorConfig = field(default_factory=lambda: DurationPredictorConfig())
    waveform_decoder:WaveformDecoderConfig = field(default_factory=lambda: WaveformDecoderConfig())
    discriminator: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig())

@dataclass
class VitsLossConfig(Coqpit):
    kl_loss_alpha:int = 1.0
    disc_loss_alpha = 1.0
    gen_loss_alpha = 1.0
    feat_loss_alpha = 1.0
    mel_loss_alpha = 45.0
    dur_loss_alpha = 1.0
    speaker_encoder_loss_alpha = 9.0

@dataclass
class VitsConfig(TrainerConfig):
    """
    General training config, here you can change the batch size and others useful parameters
    """
    dataset_config: TTSDatasetConfig = field(default_factory=lambda: TTSDatasetConfig())
    audio:AudioConfig = field(default_factory=lambda: AudioConfig())
    text:TextConfig = field(default_factory=lambda: TextConfig())

    # the max size of eval dataset
    eval_split_max_size: int = 256
    # the percentage of dataset to be eval dataset
    eval_split_size: float = 0.01

    #loss
    loss:VitsLossConfig = field(default_factory=lambda: VitsLossConfig())

    # model config
    model: VitsModelConfig = field(default_factory=lambda: VitsModelConfig())