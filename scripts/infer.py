#!/usr/bin/env python3
"""Gera legenda para uma imagem usando checkpoint treinado.

Uso:
    python scripts/infer.py --checkpoint checkpoints/epoch_5.pt --image path/to/foto.jpg

Modos:
  • img → caption          (treinado sem prompt) — gera direto
  • img + prompt → res     (treinado com prompt) — gera após o prompt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from nanoqwenvl import NanoQwenVL
from nanoqwenvl.config import load_config


def build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description="Gera legenda com NanoQwenVL")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", default=None,
                        help="Prompt opcional p/ gerar. Se null, usa o do config.yaml ou gera direto")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NanoQwenVL(config.get("model"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform()
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Se tem prompt, o modelo foi treinado para gerar resposta após o prompt
    prompt = args.prompt or config.get("data", {}).get("prompt_template")
    if prompt:
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id] + prompt_ids], dtype=torch.long, device=device
        )
        # Roda decoder uma vez com o prompt para preencher o cache de estados
        with torch.no_grad():
            text_features = model.embed_tokens(input_ids)
            visual_features = model.vit(image)
            visual_features = model.projector(visual_features)
            hidden = torch.cat([visual_features, text_features], dim=1)
            seq_len = hidden.shape[1]
            cos, sin = model.rotary_emb(seq_len, device)
            mask = torch.triu(
                torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device),
                diagonal=1,
            )
            for layer in model.layers:
                hidden = layer(hidden, cos, sin, mask)
            hidden = model.norm(hidden)
            logits = model.lm_head(hidden)[:, -1, :] / args.temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([input_ids, next_token], dim=-1)

        for _ in range(args.max_tokens - 1):
            with torch.no_grad():
                text_features = model.embed_tokens(generated)
                hidden = torch.cat([visual_features, text_features], dim=1)
                seq_len = hidden.shape[1]
                cos, sin = model.rotary_emb(seq_len, device)
                mask = torch.triu(
                    torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device),
                    diagonal=1,
                )
                for layer in model.layers:
                    hidden = layer(hidden, cos, sin, mask)
                hidden = model.norm(hidden)
                logits = model.lm_head(hidden)[:, -1, :] / args.temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                generated = torch.cat([generated, next_token], dim=-1)

        caption = tokenizer.decode(generated[0], skip_special_tokens=True)
    else:
        generated = model.generate(
            image,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
        )
        caption = tokenizer.decode(generated[0], skip_special_tokens=True)

    print("=" * 50)
    print(f"Imagem: {args.image}")
    print(f"Legenda: {caption}")
    print("=" * 50)


if __name__ == "__main__":
    main()
