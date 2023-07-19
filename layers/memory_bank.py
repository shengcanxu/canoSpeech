import torch
from torch import nn
from layers.transformer import RelativePositionMultiHeadAttention


class VAEMemoryBank(nn.Module):
    def __init__(
        self,
        bank_size=1000,
        n_hidden_dims=192,
        n_attn_heads=2,
        init_values=None
    ):
        super().__init__()

        self.bank_size = bank_size
        self.n_hidden_dims = n_hidden_dims
        self.n_attn_heads = n_attn_heads

        self.encoder = RelativePositionMultiHeadAttention(
            channels=n_hidden_dims,
            out_channels=n_hidden_dims,
            num_heads=n_attn_heads,
        )

        self.memory_bank = nn.Parameter(torch.randn(n_hidden_dims, bank_size))
        if init_values is not None:
            with torch.no_grad():
                self.memory_bank.copy_(init_values)

    def forward(self, z):
        b, _, _ = z.shape
        return self.encoder(
            z, self.memory_bank.unsqueeze(0).repeat(b, 1, 1), attn_mask=None
        )