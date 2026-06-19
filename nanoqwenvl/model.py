import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ViTConfig, ProjectorConfig, DecoderConfig, parse_model_config
from .vit import ViT
from .decoder import DecoderBlock
from .layers import RMSNorm, RotaryEmbedding


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


class NanoQwenVL(nn.Module):
    def __init__(
        self,
        config: Optional[dict] = None,
        vit_config: Optional[ViTConfig] = None,
        projector_config: Optional[ProjectorConfig] = None,
        decoder_config: Optional[DecoderConfig] = None,
    ):
        super().__init__()
        if config is not None:
            cfg = parse_model_config(config)
            vit_config = cfg["vit"]
            projector_config = cfg["projector"]
            decoder_config = cfg["decoder"]
        vit_config = vit_config or ViTConfig()
        projector_config = projector_config or ProjectorConfig()
        decoder_config = decoder_config or DecoderConfig()

        self.decoder_config = decoder_config
        num_patches = (vit_config.image_size // vit_config.patch_size) ** 2
        self.num_visual_tokens = num_patches + 1

        self.vit = ViT(vit_config)

        self.projector = nn.Sequential(
            nn.Linear(projector_config.hidden_size, projector_config.projector_hidden_size),
            nn.GELU(),
            nn.Linear(projector_config.projector_hidden_size, projector_config.hidden_size),
        )

        self.embed_tokens = nn.Embedding(
            decoder_config.vocab_size, decoder_config.hidden_size
        )
        self.rotary_emb = RotaryEmbedding(
            decoder_config.head_dim,
            decoder_config.max_position_embeddings,
            decoder_config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [DecoderBlock(decoder_config) for _ in range(decoder_config.num_layers)]
        )
        self.norm = RMSNorm(decoder_config.hidden_size)
        self.lm_head = nn.Linear(
            decoder_config.hidden_size, decoder_config.vocab_size, bias=False
        )
        self.embed_tokens.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.ndim > 1 and "cls_token" not in name and "pos_embed" not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        visual_features = self.vit(pixel_values)
        visual_features = self.projector(visual_features)
        N_v = visual_features.shape[1]

        text_features = self.embed_tokens(input_ids)
        hidden_states = torch.cat([visual_features, text_features], dim=1)
        seq_len = hidden_states.shape[1]

        cos, sin = self.rotary_emb(seq_len, hidden_states.device)
        mask = _causal_mask(seq_len, hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        B = pixel_values.shape[0]
        device = pixel_values.device

        visual_features = self.vit(pixel_values)
        visual_features = self.projector(visual_features)

        bos_id = self.decoder_config.vocab_size - 1
        input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            text_features = self.embed_tokens(input_ids)
            hidden_states = torch.cat([visual_features, text_features], dim=1)
            seq_len = hidden_states.shape[1]

            cos, sin = self.rotary_emb(seq_len, device)
            mask = _causal_mask(seq_len, device)

            for layer in self.layers:
                hidden_states = layer(hidden_states, cos, sin, mask)

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids
