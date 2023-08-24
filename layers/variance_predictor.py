import torch
from torch import nn
from torch.nn import functional as F

from layers.normalization import LayerNorm
from layers.transformer import RelativePositionMultiHeadAttention
from util.helper import sequence_mask


class VariancePredictor(nn.Module):
    def __init__(self, channels:int, condition_channels:int, kernel_size:int, n_stack:int, n_stack_in_stack:int, attention_num_head:int, dropout_p:float):
        super().__init__()

        self.conv_blocks = torch.nn.ModuleList()
        for index in range(n_stack):
            conv_block = torch.nn.ModuleList()
            for block_index in range(n_stack_in_stack):
                conv = torch.nn.Sequential()
                conv.append(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2))
                conv.append(LayerNorm(channels))
                conv.append(nn.ReLU())
                conv.append(nn.Dropout(p=dropout_p))
                conv_block.append(conv)
            self.conv_blocks.append(conv_block)

        self.pre_attention = nn.Conv1d(condition_channels, channels, 1)  # map condition_channel to channel
        self.use_pre_attention = (channels != condition_channels)
        self.attention = RelativePositionMultiHeadAttention(
            channels=channels,
            out_channels=channels,
            num_heads=attention_num_head
        )
        self.norm = LayerNorm(channels)

        self.project = nn.Conv1d(channels, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x:torch.Tensor, masks:torch.Tensor, speech_prompts:torch.Tensor, ) -> torch.Tensor:
        ''' guess the variable (duration or f0) of text(encoding) from speech prompt
        x: [Batch, Enc_d, Enc_t or Feature_t]
        speech_prompts: [Batch, Enc_d, Prompt_t]
        '''
        if self.use_pre_attention:
            speech_prompts = self.pre_attention(speech_prompts)

        for conv_block in self.conv_blocks:
            for conv in conv_block:
                x = conv(x * masks) + x

            # Attention + Dropout + Residual + Norm
            residual = x
            x = self.attention(x=x, c=speech_prompts)
            x = self.norm(x + residual)

        x = self.project(x * masks) * masks
        x = x.squeeze(1)
        return x

class DurationPredictor(VariancePredictor):
    def __init__(self, channels:int, condition_channels:int, kernel_size:int, n_stack:int, n_stack_in_stack:int, attention_num_head:int, dropout_p:float):
        super().__init__(channels, condition_channels, kernel_size, n_stack, n_stack_in_stack, attention_num_head, dropout_p)

    def forward(self, x:torch.Tensor, masks:torch.Tensor, speech_prompts:torch.Tensor) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Enc_t or Feature_t]
        speech_prompts: [Batch, Enc_d, Prompt_t]
        '''
        durations = super().forward(x=x, masks=masks, speech_prompts=speech_prompts)
        return F.softplus(durations)


class PitchPredictor(VariancePredictor):
    def __init__(self, channels:int, condition_channels:int, kernel_size:int, n_stack:int, n_stack_in_stack:int, attention_num_head:int, dropout_p:float):
        super().__init__(channels, condition_channels, kernel_size, n_stack, n_stack_in_stack, attention_num_head, dropout_p)

        self.pitch_embedding = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=1)
        self.norm = LayerNorm(channels)

    def forward(self, x:torch.Tensor, masks:torch.Tensor, speech_prompts:torch.Tensor) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Enc_t or Feature_t]
        speech_prompts: [Batch, Enc_d, Prompt_t]
        '''
        pitches = super().forward(x=x, masks=masks, speech_prompts=speech_prompts)
        pitch_embed = self.pitch_embedding(pitches.unsqueeze(1))
        pitch_embed = self.norm(pitch_embed) * masks
        return pitches, pitch_embed