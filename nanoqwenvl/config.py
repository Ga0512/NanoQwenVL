from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ViTConfig:
    patch_size: int = 16
    image_size: int = 224
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 4
    intermediate_size: int = 1024
    num_channels: int = 3


@dataclass
class ProjectorConfig:
    hidden_size: int = 256
    projector_hidden_size: int = 512


@dataclass
class DecoderConfig:
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 32
    intermediate_size: int = 512
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0


def parse_model_config(raw: dict) -> dict:
    return {
        "vit": ViTConfig(**raw.get("vit", {})),
        "projector": ProjectorConfig(**raw.get("projector", {})),
        "decoder": DecoderConfig(**raw.get("decoder", {})),
    }


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
