import logging
import os

import numpy as np
import torch

# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
from coqpit import Coqpit

from speaker.lstm_speaker import LSTMSpeakerEncoder
from speaker.resnet_speaker import ResNetSpeakerEncoder


def sequence_mask(sequence_length, max_len=None):
    """Create a sequence mask for filtering padding in a sequence tensor.
    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.
    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = sequence_length.max()
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    # B x T_max
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


def segment(x: torch.tensor, segment_indices: torch.tensor, segment_size=4, pad_short=False):
    """Segment each sample in a batch based on the provided segment indices
    Args:
        x (torch.tensor): Input tensor.
        segment_indices (torch.tensor): Segment indices.
        segment_size (int): Expected output segment size.
        pad_short (bool): Pad the end of input tensor with zeros if shorter than the segment size.
    """
    # pad the input tensor if it is shorter than the segment size
    if pad_short and x.shape[-1] < segment_size:
        x = torch.nn.functional.pad(x, (0, segment_size - x.size(2)))

    segments = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        index_start = segment_indices[i]
        index_end = index_start + segment_size
        x_i = x[i]
        if pad_short and index_end >= x.size(2):
            # pad the sample if it is shorter than the segment size
            x_i = torch.nn.functional.pad(x_i, (0, (index_end + 1) - x.size(2)))
        segments[i] = x_i[:, index_start:index_end]
    return segments


def rand_segments(x: torch.tensor, x_lengths: torch.tensor = None, segment_size=4, let_short_samples=False, pad_short=False):
    """Create random segments based on the input lengths.
    Args:
        x (torch.tensor): Input tensor.
        x_lengths (torch.tensor): Input lengths.
        segment_size (int): Expected output segment size.
        let_short_samples (bool): Allow shorter samples than the segment size.
        pad_short (bool): Pad the end of input tensor with zeros if shorter than the segment size.
    Shapes:
        - x: :math:`[B, C, T]`
        - x_lengths: :math:`[B]`
    """
    _x_lenghts = x_lengths.clone()
    B, _, T = x.size()
    if pad_short:
        if T < segment_size:
            x = torch.nn.functional.pad(x, (0, segment_size - T))
            T = segment_size
    if _x_lenghts is None:
        _x_lenghts = T
    len_diff = _x_lenghts - segment_size
    if let_short_samples:
        _x_lenghts[len_diff < 0] = segment_size
        len_diff = _x_lenghts - segment_size
    else:
        assert all(
            len_diff > 0
        ), f" [!] At least one sample is shorter than the segment size ({segment_size}). \n {_x_lenghts}"
    segment_indices = (torch.rand([B]).type_as(x) * (len_diff + 1)).long()
    ret = segment(x, segment_indices, segment_size, pad_short=pad_short)
    return ret, segment_indices

def setup_encoder_model(config: Coqpit):
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

class StandardScaler:
    """StandardScaler for mean-scale normalization with the given mean and scale values."""

    def __init__(self, mean: np.ndarray = None, scale: np.ndarray = None) -> None:
        self.mean_ = mean
        self.scale_ = scale

    def set_stats(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def reset_stats(self):
        delattr(self, "mean_")
        delattr(self, "scale_")

    def transform(self, X):
        X = np.asarray(X)
        X -= self.mean_
        X /= self.scale_
        return X

    def inverse_transform(self, X):
        X = np.asarray(X)
        X *= self.scale_
        X += self.mean_
        return X

def load_checkpoint(path:str, model:torch.nn.Module):
    assert os.path.isfile(path)
    print("loading checkpoint {}".format(path))
    checkpoint_dict = torch.load(path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model
