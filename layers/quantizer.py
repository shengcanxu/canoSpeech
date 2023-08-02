from typing import Optional, Any, Callable, Union
import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange, repeat
from layers.transformer import RelativePositionMultiHeadAttention

def default(val: Any, d: Any) -> Any:
    return val if val is not None else d

def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def laplace_smoothing(x, n_categories: int, epsilon: float=1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)

def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num, ), device=device)

    return samples[indices]

def kmeans(samples, num_clusters: int, num_iters: int=10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means,
                                                                "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


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


# class RVQQuantizer(torch.nn.Module):
#     """
#     Quantizer from hificodec: Hifi-codec: Group-residual vector quantization for high fidelity audio codec https://arxiv.org/pdf/2305.02765
#     it uses residual vector-quantizer technique, the purpose is to reduce the complexity of latent z
#     it uses groud of residual vector, compare to memory bank, it has more detail so in theory it performs better
#     normally the first layer of residual vector contains the most detail and other are residual of the first layer
#     checkpoint from: https://huggingface.co/Dongchao/AcademiCodec/tree/main
#     """
#     def __init__(self, n_code_groups:int, n_codes:int, codebook_loss_alpha:float, commitment_loss_alpha:float):
#         super(RVQQuantizer, self).__init__()
#         assert 512 % n_code_groups == 0
#
#         self.quantizer_modules1 = nn.ModuleList([
#             RVQQuantizer_module(n_codes, 512 // n_code_groups)
#             for _ in range(n_code_groups)
#         ])
#         self.quantizer_modules2 = nn.ModuleList([
#             RVQQuantizer_module(n_codes, 512 // n_code_groups)
#             for _ in range(n_code_groups)
#         ])
#         # self.config = config
#         self.codebook_loss_alpha = codebook_loss_alpha  # e.g., 1
#         self.commitment_loss_alpha = commitment_loss_alpha  # e.g., 0.25
#         self.residul_layer = 2
#         self.n_code_groups = n_code_groups
#
#     def for_one_step(self, xin, idx):
#         xin = xin.transpose(1, 2)
#         print(xin.size())
#         x = xin.reshape(-1, 512)
#         x = torch.split(x, 512 // self.n_code_groups, dim=-1)
#         min_indicies = []
#         z_q = []
#         if idx == 0:
#             for _x, m in zip(x, self.quantizer_modules1):
#                 _z_q, _min_indicies = m(_x)
#                 z_q.append(_z_q)
#                 min_indicies.append(_min_indicies)  #B * T,
#             z_q = torch.cat(z_q, -1).reshape(xin.shape)
#             # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
#             loss = self.codebook_loss_alpha * torch.mean((z_q - xin.detach()) ** 2) \
#                 + self.commitment_loss_alpha * torch.mean((z_q.detach() - xin) ** 2)
#             z_q = xin + (z_q - xin).detach()
#             z_q = z_q.transpose(1, 2)
#             return z_q, loss, min_indicies
#         else:
#             for _x, m in zip(x, self.quantizer_modules2):
#                 _z_q, _min_indicies = m(_x)
#                 z_q.append(_z_q)
#                 min_indicies.append(_min_indicies)  #B * T,
#             z_q = torch.cat(z_q, -1).reshape(xin.shape)
#             # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
#             loss = self.codebook_loss_alpha * torch.mean((z_q - xin.detach()) ** 2) \
#                 + self.commitment_loss_alpha * torch.mean((z_q.detach() - xin) ** 2)
#             z_q = xin + (z_q - xin).detach()
#             z_q = z_q.transpose(1, 2)
#             return z_q, loss, min_indicies
#
#     def forward(self, x):
#         #B, C, T
#         quantized_out = 0.0
#         residual = x
#         all_losses = []
#         all_indices = []
#         for i in range(self.residul_layer):
#             quantized, loss, indices = self.for_one_step(residual, i)  #
#             residual = residual - quantized
#             quantized_out = quantized_out + quantized
#             all_indices.extend(indices)  #
#             all_losses.append(loss)
#         all_losses = torch.stack(all_losses)
#         loss = torch.mean(all_losses)
#         return quantized_out, loss, all_indices
#
#     def embed(self, x):
#         #idx: N, T, 4
#         #print('x ', x.shape)
#         quantized_out = torch.tensor(0.0, device=x.device)
#         x = torch.split(x, 1, 2)  # split, 将最后一个维度分开, 每个属于一个index group
#         #print('x.shape ', len(x),x[0].shape)
#         for i in range(self.residul_layer):
#             ret = []
#             if i == 0:
#                 for j in range(self.n_code_groups):
#                     q = x[j]
#                     embed = self.quantizer_modules1[j]
#                     q = embed.embedding(q.squeeze(-1))
#                     ret.append(q)
#                 ret = torch.cat(ret, -1)
#                 #print(ret.shape)
#                 quantized_out = quantized_out + ret
#             else:
#                 for j in range(self.n_code_groups):
#                     q = x[j + self.n_code_groups]
#                     embed = self.quantizer_modules2[j]
#                     q = embed.embedding(q.squeeze(-1))
#                     ret.append(q)
#                 ret = torch.cat(ret, -1)
#                 quantized_out = quantized_out + ret
#         return quantized_out.transpose(1, 2)  #N, C, T
#
# class RVQQuantizer_module(torch.nn.Module):
#     """ Quantizer module used by Quantizer"""
#     def __init__(self, n_e, e_dim):
#         super(RVQQuantizer_module, self).__init__()
#         self.embedding = nn.Embedding(n_e, e_dim)
#         self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
#
#     def forward(self, x):
#         # compute Euclidean distance
#         d = torch.sum(x ** 2, 1, keepdim=True) + torch.sum(self.embedding.weight ** 2, 1) \
#             - 2 * torch.matmul(x, self.embedding.weight.T)
#         min_indicies = torch.argmin(d, 1)
#         z_q = self.embedding(min_indicies)
#         return z_q, min_indicies

class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
            self,
            dim: int,
            codebook_size: int,
            kmeans_init: int=False,
            kmeans_iters: int=10,
            decay: float=0.99,
            epsilon: float=1e-5,
            threshold_ema_dead_code: int=2, ):
        super().__init__()
        self.decay = decay
        init_fn: Union[Callable[..., torch.Tensor], Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size,
                                     self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        #broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size), self.embed)
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        #broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed +
                 embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size,
                                  self.epsilon) * self.cluster_size.sum())
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(
            self,
            dim: int,
            codebook_size: int,
            codebook_dim: Optional[int]=None,
            decay: float=0.99,
            epsilon: float=1e-5,
            kmeans_init: bool=True,
            kmeans_iters: int=50,
            threshold_ema_dead_code: int=2,
            commitment_weight: float=1., ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim)
                           if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim)
                            if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers:int, codebook_dim:int, codebook_size:int):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantization(
                dim = codebook_dim,
                codebook_size = codebook_size
            ) for _ in range(num_quantizers)
        ])

    def forward(self, x, n_q: Optional[int]=None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self,
               x: torch.Tensor,
               n_q: Optional[int]=None,
               st: Optional[int]=None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:  # 设置解码的起止layer
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
