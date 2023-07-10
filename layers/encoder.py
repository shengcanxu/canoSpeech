import math
import torch
from torch import nn

from layers.wavenet import WN
from layers.transformer import RelativePositionTransformer
from util.helper import sequence_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
    ):
        """Text Encoder for VITS model.
        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            num_heads (int): Number of attention heads for the Transformer layers.
            num_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout_p (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if language_emb_dim:
            hidden_channels += language_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        assert x.shape[0] == x_lengths.shape[0]
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]

        # concat the lang emb in embedding chars
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # [b, 1, t]

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class AudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        """Posterior Encoder of VITS model.
        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z
        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask

