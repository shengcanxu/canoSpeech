import torch
from torch import nn
from layers.transformer import RelativePositionMultiHeadAttention


class VAEMemoryBank(nn.Module):
    """
    memory bank is used in Natural Speech paper for reduce the burden of generating
    audio from text.
    it use the z generated from text as query to map to a number of parameters. use
    those parameters rather than z to generate the audio(y_hat)
    """
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
            x=z,
            c=self.memory_bank.unsqueeze(0).repeat(b, 1, 1),
            attn_mask=None
        )


class RVQQuantizer(torch.nn.Module):
    """
    Quantizer from hificodec: Hifi-codec: Group-residual vector quantization for high fidelity audio codec https://arxiv.org/pdf/2305.02765
    it uses residual vector-quantizer technique, the purpose is to reduce the complexity of latent z
    it uses groud of residual vector, compare to memory bank, it has more detail so in theory it performs better
    normally the first layer of residual vector contains the most detail and other are residual of the first layer
    checkpoint from: https://huggingface.co/Dongchao/AcademiCodec/tree/main
    """
    def __init__(self, n_code_groups:int, n_codes:int, codebook_loss_alpha:float, commitment_loss_alpha:float):
        super(RVQQuantizer, self).__init__()
        assert 512 % n_code_groups == 0

        self.quantizer_modules1 = nn.ModuleList([
            RVQQuantizer_module(n_codes, 512 // n_code_groups)
            for _ in range(n_code_groups)
        ])
        self.quantizer_modules2 = nn.ModuleList([
            RVQQuantizer_module(n_codes, 512 // n_code_groups)
            for _ in range(n_code_groups)
        ])
        # self.config = config
        self.codebook_loss_alpha = codebook_loss_alpha  # e.g., 1
        self.commitment_loss_alpha = commitment_loss_alpha  # e.g., 0.25
        self.residul_layer = 2
        self.n_code_groups = n_code_groups

    def for_one_step(self, xin, idx):
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, 512)
        x = torch.split(x, 512 // self.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        if idx == 0:
            for _x, m in zip(x, self.quantizer_modules1):
                _z_q, _min_indicies = m(_x)
                z_q.append(_z_q)
                min_indicies.append(_min_indicies)  #B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_alpha * torch.mean((z_q - xin.detach()) ** 2) \
                + self.commitment_loss_alpha * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies
        else:
            for _x, m in zip(x, self.quantizer_modules2):
                _z_q, _min_indicies = m(_x)
                z_q.append(_z_q)
                min_indicies.append(_min_indicies)  #B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_alpha * torch.mean((z_q - xin.detach()) ** 2) \
                + self.commitment_loss_alpha * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies

    def forward(self, x):
        #B, C, T
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []
        for i in range(self.residul_layer):
            quantized, loss, indices = self.for_one_step(residual, i)  #
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.extend(indices)  #
            all_losses.append(loss)
        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)
        return quantized_out, loss, all_indices

    def embed(self, x):
        #idx: N, T, 4
        #print('x ', x.shape)
        quantized_out = torch.tensor(0.0, device=x.device)
        x = torch.split(x, 1, 2)  # split, 将最后一个维度分开, 每个属于一个index group
        #print('x.shape ', len(x),x[0].shape)
        for i in range(self.residul_layer):
            ret = []
            if i == 0:
                for j in range(self.n_code_groups):
                    q = x[j]
                    embed = self.quantizer_modules1[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                #print(ret.shape)
                quantized_out = quantized_out + ret
            else:
                for j in range(self.n_code_groups):
                    q = x[j + self.n_code_groups]
                    embed = self.quantizer_modules2[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                quantized_out = quantized_out + ret
        return quantized_out.transpose(1, 2)  #N, C, T

class RVQQuantizer_module(torch.nn.Module):
    """ Quantizer module used by Quantizer"""
    def __init__(self, n_e, e_dim):
        super(RVQQuantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        # compute Euclidean distance
        d = torch.sum(x ** 2, 1, keepdim=True) + torch.sum(self.embedding.weight ** 2, 1) \
            - 2 * torch.matmul(x, self.embedding.weight.T)
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        return z_q, min_indicies