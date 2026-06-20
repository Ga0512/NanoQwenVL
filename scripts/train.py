#!/usr/bin/env python3
"""Treina o NanoQwenVL.

Formatos suportados:
  • img → caption              (sem prompt)
  • img + prompt → resposta    (com prompt_template no config.yaml)

Uso:
    python scripts/train.py --config config.yaml
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from nanoqwenvl import NanoQwenVL
from nanoqwenvl.config import load_config
from nanoqwenvl.dataset import Flickr8kHF, Flickr8kLocal, CsvCaptionDataset, DriveLMNuScenesDataset, collate_fn
from nanoqwenvl.trainer import Trainer


def build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_tokenizer():
    t = AutoTokenizer.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    return t


def create_dataset(cfg, split, transform, tokenizer):
    dc = cfg.get("data", {})
    dtype = dc.get("dataset_type", "flickr8k")
    prompt = dc.get("prompt_template")

    if dtype == "drivelm_nus":
        if split != "train":
            return None  # só tem split train
        return DriveLMNuScenesDataset(
            split=split,
            transform=transform,
            tokenizer=tokenizer,
            max_length=dc.get("max_seq_length", 128),
            max_samples=dc.get("max_samples"),
            prompt_template=prompt,
            synthetic=dc.get("synthetic", False),
            nuscenes_root=dc.get("nuscenes_root"),
        )
    elif dtype == "flickr8k_hf":
        return Flickr8kHF(
            split=split,
            transform=transform,
            tokenizer=tokenizer,
            max_length=dc.get("max_seq_length", 64),
            max_samples=dc.get("max_samples"),
            prompt_template=prompt,
            synthetic=dc.get("synthetic", False),
        )
    elif dtype == "flickr8k":
        return Flickr8kLocal(
            split=split,
            transform=transform,
            tokenizer=tokenizer,
            max_length=dc.get("max_seq_length", 64),
            image_dir=dc.get("image_dir"),
            synthetic=dc.get("synthetic", False),
            max_samples=dc.get("max_samples"),
            prompt_template=prompt,
        )
    elif dtype == "csv":
        key = "csv_train" if split == "train" else "csv_val"
        path = dc.get(key)
        if not path:
            return None
        return CsvCaptionDataset(
            csv_path=path,
            transform=transform,
            tokenizer=tokenizer,
            max_length=dc.get("max_seq_length", 64),
            prompt_template=prompt,
        )
    else:
        raise ValueError(f"dataset_type desconhecido: {dtype}")


def main():
    parser = argparse.ArgumentParser(description="Treina NanoQwenVL")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NanoQwenVL(config.get("model"))
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {num_params:,}")

    transform = build_transform()
    tokenizer = build_tokenizer()
    pad_token_id = tokenizer.pad_token_id

    train_ds = create_dataset(config, "train", transform, tokenizer)
    val_ds = create_dataset(config, "validation", transform, tokenizer)

    if train_ds is None:
        print("Nenhum dataset configurado. Verifique config.yaml")
        return

    prompt_label = config.get("data", {}).get("prompt_template", "null")
    print(f"Modo: {'img + \"' + prompt_label + '\" → resposta' if prompt_label else 'img → caption'}")
    if hasattr(train_ds, "num_prompt_tokens"):
        print(f"   num_prompt_tokens (fixo): {train_ds.num_prompt_tokens}")
    print(f"Train: {len(train_ds)} samples")

    collate = lambda b: collate_fn(b, pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=config["training"].get("batch_size", 16),
        shuffle=True, num_workers=0, collate_fn=collate,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        print(f"Val: {len(val_ds)} samples")
        val_loader = DataLoader(
            val_ds, batch_size=config["training"].get("batch_size", 16),
            shuffle=False, num_workers=0, collate_fn=collate,
        )

    trainer_cfg = dict(config["training"])
    trainer_cfg["pad_token_id"] = pad_token_id
    if hasattr(train_ds, "num_prompt_tokens"):
        trainer_cfg["num_prompt_tokens"] = train_ds.num_prompt_tokens
    trainer = Trainer(model, trainer_cfg, device)
    trainer.fit(train_loader, val_loader)

    print("Treino concluído!")


if __name__ == "__main__":
    main()
