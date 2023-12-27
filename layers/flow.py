import math
import torch
from torch import nn

from layers.transformer import RelativePositionMultiHeadAttention, RelativePositionTransformer
from layers.wavenet import WaveNet


class FilM(nn.Module):
    """Figure 4 in NaturalSpeech2 paper"""
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels= condition_channels,
            out_channels= channels * 2,
            kernel_size= 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        masks: torch.Tensor,
    ):
        conditions = self.conv(conditions) * masks
        betas, gammas = conditions.chunk(chunks= 2, dim= 1)
        x = gammas * x + betas

        return x * masks


class AttentionFlowBlock(nn.Module):
    """
    use wavenet and attention. As descripted in Figure 4 in NaturalSpeech2 paper, but change a little bit.
    the changes is:
        1. keep the flow architecture from VITS
        2. wrape the wavenet, add attention and FiLM to it, as description in Figure 4
    """
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
        attention_heads=2,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.wavenet = WaveNet(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        self.attention = RelativePositionMultiHeadAttention(
            channels=hidden_channels,
            out_channels=hidden_channels,
            num_heads=attention_heads
        )
        self.film = FilM(
            channels=hidden_channels,
            condition_channels=hidden_channels
        )

        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, condition=None, reverse=False):
        """Note:
            Set `reverse` to True for inference.
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask

        # major operation in flow. we can use any operations here (on one half of channels
        pre_wavenet = h
        h = self.wavenet(h, x_mask, g=g)
        if condition is not None:
            condition = self.attention(x=pre_wavenet, c=condition)
            h = self.film(x=h, conditions=condition, masks=x_mask)

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class AttentionFlow(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
        attention_heads=2
    ):
        """Redisual Coupling blocks for VITS flow layers.
        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels.
            kernel_size (int): Kernel size of the WaveNet layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            num_flows (int, optional): Number of Residual Coupling blocks. Defaults to 4.
            cond_channels (int, optional): Number of channels of the conditioning tensor. Defaults to 0.
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                AttentionFlowBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=True,
                    attention_heads=attention_heads
                )
            )

    def forward(self, x, x_mask, g=None, condition=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, condition=condition, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, condition=condition, reverse=reverse)
        return x


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
        use_transformer_flow=False,
        use_SNAC=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        #vits2: transformer in each flow block
        self.pre_transformer = None
        if use_transformer_flow and not use_SNAC:
            self.pre_transformer = RelativePositionTransformer(
                in_channels=self.half_channels,
                out_channels=self.half_channels,
                hidden_channels=self.half_channels,
                hidden_channels_ffn=self.half_channels,
                num_heads=2,
                num_layers=1,
                kernel_size=3,
                dropout_p=0.1,
                rel_attn_window_size=None,
            )

        # SNAC layer to add speakering information
        self.use_SNAC = use_SNAC
        if self.use_SNAC:
            self.sn_linear = nn.Conv1d(cond_channels, 2*self.half_channels, 1)

        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WaveNet(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """Note:
            Set `reverse` to True for inference.
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if self.use_SNAC and g is not None:
            return self.do_SNAC_forward(x, x_mask, g=g, reverse=reverse)
        else:
            return self.do_forward(x, x_mask, g=g, reverse=reverse)

    def do_forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        #vits2: transformer in each flow
        x0_pre = x0
        if self.pre_transformer is not None:
            x0_pre = self.pre_transformer(x0 * x_mask, x_mask)
            x0_pre = x0_pre + x0  # residual connection

        h = self.pre(x0_pre) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def do_SNAC_forward(self, x, x_mask, g=None, reverse=False):
        sn_variables = self.sn_linear(g)
        sn_m, sn_v = sn_variables.chunk(2, dim=1)  # (B, half_chnnael, 1)
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        #* Pass x0 to SN before WN
        h = (x0 - sn_m) * torch.exp(-sn_v) * x_mask
        h = self.pre(h) * x_mask
        #* Global conditioning is not used
        h = self.enc(h, x_mask, g=None)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)  #logs (s) are fixed to 0

        if not reverse:
            #* SN to x1 before affine xform
            x1 = (x1 - sn_m) * torch.exp(-sn_v) * x_mask
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs * x_mask, [1,2]) - torch.sum(sn_v.expand(-1,-1,logs.size(-1)) * x_mask, [1,2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            #* SDN before concat
            x1 = (sn_m + x1 * torch.exp(sn_v)) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
        use_transformer_flow=False,
        use_SNAC=False,
    ):
        """Redisual Coupling blocks for VITS flow layers.
        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels.
            kernel_size (int): Kernel size of the WaveNet layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            num_flows (int, optional): Number of Residual Coupling blocks. Defaults to 4.
            cond_channels (int, optional): Number of channels of the conditioning tensor. Defaults to 0.
            use_transformer_flow: add transformer in flow
            use_SNAC: use SNAC in flow layer. transformer is not used
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=True,
                    use_transformer_flow=use_transformer_flow,
                    use_SNAC=use_SNAC
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        total_logdet = 0
        if not reverse:
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
                total_logdet += log_det
            return x, total_logdet
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
            return x
