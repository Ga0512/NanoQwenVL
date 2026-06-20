# NanoQwenVL

A Vision-Language Model built from scratch in pure PyTorch for learning purposes.

**22M parameters** • **Fits on a T4 (6-16GB)** • **No heavy dependencies**

## Architecture

```
Image ─→ ViT (patch 16, 6 blocks, 4 heads) ─→ Projector (MLP 256→512→256)
                                                         │
                                                   [visual tokens 197]
                                                         │
Prompt ─→ Tokenizer ─→ [BOS, prompt...] ─→ Embedding ─── concat ─→ Decoder
                                                                     │
Response ←── LM Head ←─── RMSNorm ←─── DecoderBlock ×6 ←─── RoPE ─── causal mask
                                         │
                                    SwiGLU FFN (8 heads, head_dim=32)
```

| Component | Details |
|---|---|
| **ViT** | Patch size 16, 224×224 → 197 tokens, 6 blocks, RMSNorm, GELU MLP |
| **Projector** | MLP 256 → 512 → 256 |
| **Decoder** | 6 layers, 8 heads (head_dim=32), RoPE (θ=10000), RMSNorm, SwiGLU, weight tying |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision
- transformers
- datasets
- PyYAML
- tqdm
- Pillow

## DriveLM-nuScenes (Colab T4)

Open [Google Colab](https://colab.research.google.com), select a **T4 GPU** runtime.

### 1. Clone and install

```python
!git clone https://github.com/anomalyco/NanoQwenVL.git
%cd NanoQwenVL
!pip install -e .
```

### 2. Download nuScenes images

As anotações (5280 QA pairs) são carregadas automaticamente do HuggingFace (`ac4462/DriveLM-reasoning`). Você só precisa das imagens nuScenes.

**Opção A —Subset oficial DriveLM (~1.6 GB, recomendado):**

Preencha o [formulário do DriveLM](https://docs.google.com/forms/d/e/1FAIpQLSeX6CR3u-15IV-TKx2uPv1wiKjydjZ__NNW98H4nR5JZtQa2Q/viewform) e baixe `drivelm_nus_imgs_train.zip`. Depois:

```python
from google.colab import files
# upload do zip
!unzip -q drivelm_nus_imgs_train.zip -d /content/nuscenes
```

**Opção B — nuScenes completo (license-free para pesquisa):**

Registre em [nuscenes.org](https://www.nuscenes.org) e baixe a pasta `samples/`. Coloque em `/content/nuscenes/samples/`.

### 3. Train (30 épocas, ~15 min no T4)

```python
!python scripts/train.py --config config.yaml
```

config.yaml já vem pronto:
- `dataset_type: drivelm_nus`
- `nuscenes_root: /content/nuscenes/samples`
- `batch_size: 32`, `num_epochs: 30`

Expected:
```
Device: cuda
Params: 22,047,488
Train: 5280 samples
Epoch 1/30  train loss=10.9  ppl=54000
...
Epoch 30/30  train loss=2.1  ppl=8.2
```

### 4. Test

```python
import os
imgs = os.listdir("/content/nuscenes/samples/CAM_FRONT")
test_img = os.path.join("/content/nuscenes/samples/CAM_FRONT", imgs[0])
```

```python
!python scripts/infer.py \
  --checkpoint checkpoints/epoch_30.pt \
  --image "{test_img}" \
  --max-tokens 64
```

---

## Synthetic test (no images needed)

```bash
python scripts/train.py --config config.yaml
```

Antes de rodar, mude no config.yaml:
```yaml
data:
  dataset_type: flickr8k_hf
  synthetic: true
```

---

## Using another dataset

### CSV format

```yaml
data:
  dataset_type: csv
  csv_train: /path/to/train.tsv
  csv_val: /path/to/val.tsv
```

Formato: `image_path\t caption` por linha.

### Custom class

```python
from nanoqwenvl import ImageCaptionDataset

class MeuDataset(ImageCaptionDataset):
    def __init__(self, transform, tokenizer, max_seq_length, ...):
        self.samples = [(img_path, pergunta, resposta), ...]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ...
        return image_tensor, token_ids
```

Registre em `scripts/train.py → create_dataset()`.

### HuggingFace datasets

```yaml
data:
  dataset_type: flickr8k_hf  # jxie/flickr8k
```

---

## Inference

```bash
# img → caption (prompt_template: null)
python scripts/infer.py \
  --checkpoint checkpoints/epoch_30.pt \
  --image /path/to/photo.jpg

# img + prompt → resposta (prompt_template definido)
python scripts/infer.py \
  --checkpoint checkpoints/epoch_30.pt \
  --image /path/to/photo.jpg \
  --prompt "What is shown in this image?"
```

---

## Parameter count (22M)

| Submodel | Params |
|---|---|
| ViT | 4.9M |
| Projector | 262K |
| Embedding + LM Head | 12.8M (tied) |
| Decoder (6× blocks) | 3.9M |
| **Total** | **22.0M** |

fp32: ~88 MB. With AdamW optimizer: ~700 MB. Fits on any GPU ≥ 6GB.

## License

MIT
