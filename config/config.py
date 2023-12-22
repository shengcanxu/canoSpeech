from dataclasses import dataclass, field, asdict
from typing import List, Union

from coqpit import Coqpit, check_argument
from trainer import TrainerConfig

@dataclass
class DatasetInfoConfig(Coqpit):
    dataset_name: str = ""
    path: str = ""
    meta_file_train: str = ""
    meta_file_val: str = ""
    language: str = "en"

@dataclass
class TTSDatasetConfig(Coqpit):
    datasets:List[DatasetInfoConfig] = field(default_factory=lambda: [DatasetInfoConfig()])

    use_cache: bool = False
    num_loader_workers:int = 8
    num_eval_loader_workers:int = 8
    add_preprocess_data:bool = True

@dataclass
class TextConfig(Coqpit):
    min_text_len:int = 1
    max_text_len:int = 190
    add_blank:bool = True
    cleaned_text:bool = True
    text_cleaners:dict = field(default_factory=lambda: {"en": "english_cleaners2"})

@dataclass
class AudioConfig(Coqpit):
    mel_fmin: float = 0.0
    mel_fmax = None
    hop_length:int = 256
    win_length:int = 1024
    sample_rate:int = 22050  # sample_rate affect the training time a lot
    fft_size:int = 1024
    num_mels:int = 80
    pitch_fmax:float = 640.0
    pitch_fmin:float = 1.0
    max_audio_length:float = 10.0
    min_audio_length:float = 1.0
    preemphasis: float = 0.0

@dataclass
class TextEncoderConfig(Coqpit):
    hidden_channels_ffn:int = 768
    num_heads:int = 2
    num_layers:int = 10
    kernel_size:int = 3
    dropout_p:float = 0.1

@dataclass
class AudioEncoderConfig(Coqpit):
    kernel_size:int = 5
    dilation_rate:int = 1
    num_layers:int = 16

@dataclass
class FlowConfig(Coqpit):
    kernel_size:int = 5
    dilation_rate:int = 1
    num_flows:int = 4
    num_layers_in_flow:int = 4
    attention_heads:int = 2

@dataclass
class VitsDurationPredictorConfig(Coqpit):
    kernel_size:int = 3
    filter_channels:int = 256
    use_stochastic_dp:bool = True
    dropout_p:float = 0.5

@dataclass
class WaveformDecoderConfig(Coqpit):
    resblock_type:str = "1"
    resblock_dilation_sizes:List[List[int]] = field(default_factory=lambda: [
                [ 1, 3, 5 ],
                [ 1, 3, 5 ],
                [ 1, 3, 5 ]
            ])
    resblock_kernel_sizes:List[int] = field(default_factory=lambda: [ 3, 7, 11])
    upsample_kernel_sizes:List[int] = field(default_factory=lambda: [ 16, 16, 4, 4])
    upsample_initial_channel:int = 512
    upsample_rates:List[int] = field(default_factory=lambda: [ 8, 8, 2, 2])

@dataclass
class DiscriminatorConfig(Coqpit):
    periods_multi_period:List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_spectral_norm:bool = False

@dataclass
class LearnableUpsampling(Coqpit):
    d_predictor:int = 192
    kernel_size_lu:int = 3
    dropout_lu:float = 0.0
    conv_output_size:int = 8
    dim_w:int = 4
    dim_c:int = 2
    max_seq_len:int = 1000  # max sequence length

@dataclass
class MemroyBank(Coqpit):
    bank_size:int = 1000
    n_hidden_dims:int = 192
    n_attn_heads:int = 2

@dataclass
class BaseModelConfig(Coqpit):
    hidden_channels: int = 192
    out_channels:int = 513
    spec_segment_size:int = 32

    use_sdp: bool = True
    language_embedding_channels: int = 4
    use_language_ids:bool = False
    speaker_embedding_channels:int = 512
    use_speaker_ids:bool = False
    use_speaker_embeds:bool = False

    use_speaker_encoder_as_loss:bool = False

@dataclass
class ModelConfig(BaseModelConfig):
    use_memory_bank: bool = True
    use_gt_duration: bool = False  # use ground-true duration to generate the training data

    text_encoder: TextEncoderConfig = field(default_factory=lambda: TextEncoderConfig())
    audio_encoder: AudioEncoderConfig = field(default_factory=lambda: AudioEncoderConfig())
    flow:FlowConfig = field(default_factory=lambda:FlowConfig())
    duration_predictor:VitsDurationPredictorConfig = field(default_factory=lambda: VitsDurationPredictorConfig())
    waveform_decoder:WaveformDecoderConfig = field(default_factory=lambda: WaveformDecoderConfig())
    discriminator: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig())

    learnable_upsampling: LearnableUpsampling = field(default_factory=lambda: LearnableUpsampling())
    memory_bank: MemroyBank = field(default_factory=lambda: MemroyBank())

@dataclass
class LossConfig(Coqpit):
    kl_loss_alpha:float = 1.0
    kl_loss_forward_alpha:float = 1.0
    disc_loss_alpha:float = 1.0
    gen_loss_alpha:float = 1.0
    gen_loss_e2e_alpha:float = 1.0
    feat_loss_alpha:float = 1.0
    mel_loss_alpha:float = 45.0
    dur_loss_alpha:float = 1.0
    pitch_loss_alpha:float = 1.0
    speaker_encoder_loss_alpha:float = 9.0

    use_soft_dynamic_time_warping:bool = False

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
    # path for continue training
    continue_path: str = None
    restore_path: str = None

    #loss
    loss:LossConfig = field(default_factory=lambda: LossConfig())

    # model config
    model: ModelConfig = field(default_factory=lambda: ModelConfig())


################################ Natural TTS ##########################################

@dataclass
class QuantizerConfig(Coqpit):
    num_quantizers:int = 8
    codebook_size:int = 1024
    codebook_dimension:int = 192
    codebook_loss_alpha:float = 1.0

@dataclass
class PitchPredictorConfig(Coqpit):
    kernel_size:int = 3
    n_stack:int = 10
    n_stack_in_stack:int = 3
    attention_num_head:int = 2
    dropout_p:float = 0.5

@dataclass
class DurationPredictorConfig(Coqpit):
    kernel_size:int = 3
    n_stack:int = 10
    n_stack_in_stack:int = 3
    attention_num_head:int = 2
    dropout_p:float = 0.5

@dataclass
class NaturalTTSModelConfig(BaseModelConfig):
    use_gt_duration:bool = False  # use ground-true duration to generate the training data

    text_encoder: TextEncoderConfig = field(default_factory=lambda: TextEncoderConfig())
    audio_encoder: AudioEncoderConfig = field(default_factory=lambda: AudioEncoderConfig())
    flow:FlowConfig = field(default_factory=lambda:FlowConfig())
    duration_predictor:DurationPredictorConfig = field(default_factory=lambda: DurationPredictorConfig())
    pitch_predictor:PitchPredictorConfig = field(default_factory=lambda: PitchPredictorConfig())
    learnable_upsampling:LearnableUpsampling = field(default_factory=lambda: LearnableUpsampling())
    waveform_decoder:WaveformDecoderConfig = field(default_factory=lambda: WaveformDecoderConfig())
    quantizer:QuantizerConfig = field(default_factory=lambda: QuantizerConfig())
    discriminator: DiscriminatorConfig = field(default_factory=lambda: DiscriminatorConfig())

@dataclass
class NaturalTTSConfig(TrainerConfig):
    """
    General training config, here you can change the batch size and others useful parameters
    """
    dataset_config: TTSDatasetConfig = field(default_factory=lambda: TTSDatasetConfig())
    audio: AudioConfig = field(default_factory=lambda: AudioConfig())
    text: TextConfig = field(default_factory=lambda: TextConfig())

    # the max size of eval dataset
    eval_split_max_size: int = 256
    # the percentage of dataset to be eval dataset
    eval_split_size: float = 0.01
    # path for continue training
    continue_path: str = None
    restore_path: str = None

    # loss
    loss: LossConfig = field(default_factory=lambda: LossConfig())

    # model config
    model: NaturalTTSModelConfig = field(default_factory=lambda: NaturalTTSModelConfig())
