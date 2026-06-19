import torch
import torch.nn as nn

from .config import DecoderConfig
from .layers import RMSNorm, SwiGLU, RotaryEmbedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DecoderAttention(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        hidden_size = config.hidden_size

        self.q_proj = nn.Linear(hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, config.num_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos[:T], sin[:T])

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn + mask
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = DecoderAttention(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.mlp(self.norm2(x))
        return x
