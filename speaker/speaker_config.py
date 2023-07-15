from dataclasses import dataclass, field
from coqpit import Coqpit

from config.config import AudioConfig

@dataclass
class ModelParamsConfig(Coqpit):
    model_name: str = "resnet"
    input_dim: int = 64
    use_torch_spec:bool = True
    log_input:bool = True
    proj_dim:int = 512

@dataclass
class SpeakerConfig(Coqpit):
    audio:AudioConfig = field(default_factory=lambda: AudioConfig())
    model_params:ModelParamsConfig = field(default_factory=lambda: ModelParamsConfig())