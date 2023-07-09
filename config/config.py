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

    formatter: str = ""
    dataset_name: str = ""
    path: str = ""
    meta_file_train: str = ""
    meta_file_val: str = ""
    ignored_speakers: List[str] = None
    language: str = "en"
    # phonemizer: str = ""

    mel_fmin: float = 0
    mel_fmax = None
    hop_length:int = 256
    win_length:int = 1024
    sample_rate:int = 16000
    fft_length:int = 1024
    num_mels:int = 80

    min_text_len:int = 1
    max_text_len:int = 190
    add_blank:bool = True
    cleaned_text:bool = True
    text_cleaners:List[str] = None

@dataclass
class DiscriminatorConfig(Coqpit):
    periods_multi_period_discriminator:List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_spectral_norm_disriminator:bool = False

@dataclass
class TextEncoderConfig(Coqpit):
    num_chars:int = 165
    hidden_channels_ffn_text_encoder:int = 768
    num_heads_text_encoder:int = 2
    num_layers_text_encoder:int = 10
    kernel_size_text_encoder:int = 3
    dropout_p_text_encoder:int = 0.1

@dataclass
class AudioEncoderConfig(Coqpit):
    kernel_size_audio_encoder:int = 5
    dilation_rate_audio_encoder:int = 1
    num_layers_audio_encoder:int = 16

@dataclass
class TTSModelConfig(Coqpit):
    hidden_channels: int = 192
    out_channels:int = 513
    embedded_language_dim: int = 4
    embedded_speaker_dim:int = 256
    discriminator: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig())
    text_encoder: TextEncoderConfig = field(default_factory=lambda: TextEncoderConfig())
    audio_encoder: AudioEncoderConfig = field(default_factory=lambda: AudioEncoderConfig())

@dataclass
class TrainTTSConfig(TrainerConfig):
    """
    General training config, here you can change the batch size and others useful parameters
    """
    # dataset config
    dataset_config: TTSDatasetConfig = field(default_factory=lambda: TTSDatasetConfig())

    # the max size of eval dataset
    eval_split_max_size: int = 256
    # the percentage of dataset to be eval dataset
    eval_split_size: float = 0.01

    kl_loss_alpha:int = 1.0
    disc_loss_alpha = 1.0
    gen_loss_alpha = 1.0
    feat_loss_alpha = 1.0
    mel_loss_alpha = 45.0
    dur_loss_alpha = 1.0
    speaker_encoder_loss_alpha = 9.0

    # model config
    model: TTSModelConfig = field(default_factory=lambda: TTSModelConfig())