import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class LearnableUpsampling(nn.Module):
    def __init__(
        self,
        d_predictor=192,
        kernel_size=3,
        dropout=0.0,
        conv_output_size=8,
        dim_w=4,
        dim_c=2,
        max_seq_len=1000,
    ):
        super(LearnableUpsampling, self).__init__()
        self.max_seq_len = max_seq_len

        # Attention (W)
        self.conv_w = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_w = SwishBlock(conv_output_size + 2, dim_w, dim_w)
        self.linear_w = LinearNorm(dim_w * d_predictor, d_predictor, bias=True)
        self.softmax_w = nn.Softmax(dim=2)

        # Auxiliary Attention Context (C)
        self.conv_c = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_c = SwishBlock(conv_output_size + 2, dim_c, dim_c)

        # Upsampled Representation (O)
        self.linear_einsum = LinearNorm(dim_c * dim_w, d_predictor)  # A
        self.layer_norm = nn.LayerNorm(d_predictor)

        self.proj_o = LinearNorm(192, 192 * 2)

    def forward(self, duration, tokens, src_len, src_mask, tgt_len, max_src_len):

        batch_size = duration.shape[0]

        # Duration Interpretation
        # if tgt_len is None:
        #     mel_len = torch.round(duration.sum(-1)).type(torch.LongTensor).to(tokens.device)
        # else:
        #     # fix: mel_len should be the target_lens(y_lengths)
        #     mel_len = torch.round(tgt_len.type(torch.LongTensor).to(tokens.device))

        mel_len = torch.round(duration.sum(-1)).type(torch.LongTensor).to(tokens.device)

        mel_len = torch.clamp(mel_len, max=self.max_seq_len)
        max_mel_len = mel_len.max().item()
        mel_mask = self.get_mask_from_lengths(mel_len, max_mel_len)

        # Prepare Attention Mask
        src_mask_ = src_mask.unsqueeze(1).expand(
            -1, mel_mask.shape[1], -1
        )  # mel_mask_:[B,specT,T]
        mel_mask_ = mel_mask.unsqueeze(-1).expand(
            -1, -1, src_mask.shape[1]
        )  # mel_mask_:[B,specT,T]
        attn_mask = torch.zeros(
            (src_mask.shape[0], mel_mask.shape[1], src_mask.shape[1])
        ).to(tokens.device)  # attn_mask:[B,specT,T]
        attn_mask = attn_mask.masked_fill(src_mask_, 1.0)
        attn_mask = attn_mask.masked_fill(mel_mask_, 1.0)
        attn_mask = attn_mask.bool()

        # Token Boundary Grid
        e_k = torch.cumsum(duration, dim=1)
        s_k = e_k - duration
        e_k = e_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        s_k = s_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)  # e_k:[B,specT,T] s_k:[B,specT,T]
        t_arange = (
            torch.arange(1, max_mel_len + 1, device=tokens.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, max_src_len)
        )

        # formulate (6) in page 15 in natural speech paper
        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(
            attn_mask, 0
        )  # S:[B,specT,T] E:[B,specT,T]

        # Attention (W), formulate (7)
        W = self.swish_w(S, E, self.conv_w(tokens))  # W:[B,specT,T,dim_w]
        W = W.masked_fill(src_mask_.unsqueeze(-1), -np.inf)
        W = self.softmax_w(W)
        W = W.masked_fill(mel_mask_.unsqueeze(-1), 0.0)
        W = W.permute(0, 3, 1, 2)

        # Auxiliary Attention Context (C),  formulate (8)
        C = self.swish_c(S, E, self.conv_c(tokens))  # C:[B,specT,T,dim_c]

        # Upsampled Representation (O),  formulate (9)
        upsampled_rep = self.linear_w(
            torch.einsum("bqtk,bkh->bqth", W, tokens).permute(0, 2, 1, 3).flatten(2)
        ) + self.linear_einsum(
            torch.einsum("bqtk,btkp->bqtp", W, C).permute(0, 2, 1, 3).flatten(2)
        )  # [B,specT,C]
        upsampled_rep = self.layer_norm(upsampled_rep)
        upsampled_rep = upsampled_rep.masked_fill(mel_mask.unsqueeze(-1), 0)
        upsampled_rep = self.proj_o(upsampled_rep)  # upsampled_rep:[B,specT,C*2]

        p_mask = ~mel_mask
        return upsampled_rep, p_mask, mel_len, W, C

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (
            torch.arange(0, max_len)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(lengths.device)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask

class LinearNorm(nn.Module):
    """LinearNorm Projection"""

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional Block"""

    def __init__(
        self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()
    ):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
            ),
            nn.BatchNorm1d(out_channels),
            activation,
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, enc_input, mask=None):
        enc_output = enc_input.contiguous().transpose(1, 2)
        enc_output = F.dropout(self.conv_layer(enc_output), self.dropout, self.training)

        enc_output = self.layer_norm(enc_output.contiguous().transpose(1, 2))
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """1D Convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal

class SwishBlock(nn.Module):
    """Swish Block"""

    def __init__(self, in_channels, hidden_dim, out_channels):
        super(SwishBlock, self).__init__()
        self.layer = nn.Sequential(
            LinearNorm(in_channels, hidden_dim, bias=True),
            nn.SiLU(),
            LinearNorm(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            LinearNorm(hidden_dim, out_channels, bias=True),
        )

    def forward(self, S, E, V):
        out = torch.cat(
            [
                S.unsqueeze(-1),
                E.unsqueeze(-1),
                V.unsqueeze(1).expand(-1, E.size(1), -1, -1),
            ],
            dim=-1,
        )
        out = self.layer(out)

        return out
