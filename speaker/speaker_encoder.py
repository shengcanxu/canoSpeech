from typing import Union, List
import numpy as np
import torch
from coqpit import Coqpit
from speaker.lstm_speaker import LSTMSpeakerEncoder
from speaker.resnet_speaker import ResNetSpeakerEncoder
from speaker.speaker_config import SpeakerConfig
from util.mel_processing import wav_to_mel

class SpeakerEncoder(object):
    def __init__(
        self,
        config_path:str = "",
        model_path:str = "",
        use_cuda: bool = False
    ):
        self.config = SpeakerConfig()
        self.config.load_json(config_path)
        self.encoder = self.get_encoder_model(self.config)
        self.encoder.load_checkpoint(
            self.config,
            model_path,
            eval=True,
            use_cuda=use_cuda,
            cache=True
        )
        # self.audio_processor = AudioProcessor(**self.config.audio, verbose=False)
        self.use_cuda = use_cuda

    def get_encoder_model(self, config: Coqpit):
        if config.model_params["model_name"].lower() == "lstm":
            model = LSTMSpeakerEncoder(
                config.model_params["input_dim"],
                config.model_params["proj_dim"],
                config.model_params["lstm_dim"],
                config.model_params["num_lstm_layers"],
                use_torch_spec=config.model_params.get("use_torch_spec", False),
                audio_config=config.audio,
            )
        elif config.model_params["model_name"].lower() == "resnet":
            model = ResNetSpeakerEncoder(
                input_dim=config.model_params["input_dim"],
                proj_dim=config.model_params["proj_dim"],
                log_input=config.model_params.get("log_input", False),
                use_torch_spec=config.model_params.get("use_torch_spec", False),
                audio_config=config.audio,
            )
        return model

    def compute_embeddings(self, feats: Union[torch.Tensor, np.ndarray]) -> List:
        """Compute embedding from features.
        Args:
            feats (Union[torch.Tensor, np.ndarray]): Input features.
        Returns:
            List: computed embedding.
        """
        if isinstance(feats, np.ndarray):
            feats = torch.FloatTensor(feats)
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        if self.use_cuda:
            feats = feats.cuda()
        return self.encoder.compute_embedding(feats)

    def compute_embedding_from_waveform(self, waveform:torch.Tensor) -> list:
        """Compute a embedding from a given audio file.
        """
        if not self.config.model_params.get("use_torch_spec", False):
            # m_input = self.audio_processor.melspectrogram(waveform)
            # m_input = torch.FloatTensor(m_input)
            m_input = wav_to_mel(
                y=waveform,
                n_fft=self.config.audio.fft_size,
                sample_rate=self.config.audio.sample_rate,
                num_mels=self.config.audio.num_mels,
                hop_length=self.config.audio.hop_length,
                win_length=self.config.audio.win_length,
                fmin=self.config.audio.mel_fmin,
                fmax=self.config.audio.mel_fmax,
                center=False,
            )
        else:
            m_input = torch.FloatTensor(waveform)
        if self.use_cuda:
            m_input = m_input.cuda()
        m_input = m_input.unsqueeze(0)
        embedding = self.encoder.compute_embedding(m_input)
        return embedding