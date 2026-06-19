# NanoQwenVL

Vision-Language Model implementado do zero em PyTorch puro para estudo.

**22M parâmetros** • **Cabe numa T4 (6-16GB)** • **Sem dependências pesadas**

## Arquitetura

```
Imagem ─→ ViT (patch 16, 6 blocks, 4 heads) ─→ Projector (MLP 256→512→256)
                                                          │
                                                    [visual tokens 197]
                                                          │
Prompt ─→ Tokenizer ─→ [BOS, prompt...] ─→ Embedding ─── concat ─→ Decoder
                                                                      │
Resposta ←─ LM Head ←─── RMSNorm ←─── DecoderBlock ×6 ←─── RoPE ─── causal mask
                                          │
                                     SwiGLU FFN (8 heads, head_dim=32)
```

| Componente | Detalhes |
|---|---|
| **ViT** | Patch size 16, 224×224 → 197 tokens, 6 blocks, RMSNorm, GELU MLP |
| **Projector** | MLP 256 → 512 → 256 |
| **Decoder** | 6 layers, 8 heads (head_dim=32), RoPE (θ=10000), RMSNorm, SwiGLU, weight tying |

### Modos de treino

| Modo | config.yaml | Exemplo |
|---|---|---|
| img → caption | `prompt_template: null` | BOS + legenda |
| img + prompt → resposta | `prompt_template: "Describe this image:"` | BOS + prompt + legenda |

## Requisitos

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision
- transformers
- PyYAML
- tqdm
- Pillow

## Quick start

```bash
git clone https://github.com/seu-usuario/NanoQwenVL.git
cd NanoQwenVL
pip install -e .
```

### Treino no Colab

Abra o Colab e execute célula por célula:

```python
# 1. Clonar e instalar
!git clone https://github.com/seu-usuario/NanoQwenVL.git
%cd NanoQwenVL
!pip install -e .

# 2. Baixar Flickr8k
!wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
!unzip -q Flickr8k_text.zip -d /tmp/flickr8k/text
# As imagens precisam ser baixadas separadamente
# (Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k)

# 3. Configurar (opcional)
# Edite config.yaml para ajustar batch_size, synthetic, prompt_template etc.

# 4. Treinar
!python scripts/train.py --config config.yaml

# 5. Inferência
!python scripts/infer.py --checkpoint checkpoints/epoch_5.pt --image /path/to/foto.jpg
```

### Treino local com dados sintéticos (teste rápido)

```bash
python scripts/train.py --config config.yaml
```

Por padrão (`synthetic: true`), gera imagens aleatórias com as legendas reais do Flickr8k. Ótimo para validar o pipeline.

### Treino com imagens reais

1. Baixe o [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) (~1GB)
2. Extraia as imagens para uma pasta
3. Edite `config.yaml`:

```yaml
data:
  synthetic: false
  image_dir: /caminho/Flickr8k_Dataset
```

4. Treine:

```bash
python scripts/train.py --config config.yaml
```

## Configuração (`config.yaml`)

```yaml
model:
  vit:
    hidden_size: 256
    num_layers: 6
    num_heads: 4
  decoder:
    hidden_size: 256
    num_layers: 6
    num_heads: 8
    intermediate_size: 512

data:
  dataset_type: flickr8k          # flickr8k | csv
  synthetic: true                  # true = imagens aleatórias
  prompt_template: "Describe this image:"  # null = img→caption
  max_samples: 500                 # null = dataset completo

training:
  batch_size: 16
  learning_rate: 0.0003
  num_epochs: 10
  checkpoint_dir: checkpoints
```

## Usar outro dataset

### Opção 1: CSV

Crie um arquivo TSV com duas colunas: `caminho_da_imagem\t legenda`:

```yaml
data:
  dataset_type: csv
  csv_train: data/train.tsv
  csv_val: data/val.tsv
```

### Opção 2: Classe própria

```python
from nanoqwenvl import ImageCaptionDataset

class MeuDataset(ImageCaptionDataset):
    def __init__(self, ..., prompt_template=None):
        # Carregue seus dados
        if prompt_template:
            self.prompt_ids = tokenizer.encode(prompt_template)
            self.num_prompt_tokens = 1 + len(self.prompt_ids)
        else:
            self.num_prompt_tokens = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retorna (image_tensor, token_ids)
        return image, self._tokenize(caption)
```

Depois registre no `scripts/train.py` na função `create_dataset()`.

## Inferência

```bash
# Modo img → caption (treinado sem prompt)
python scripts/infer.py \
  --checkpoint checkpoints/epoch_10.pt \
  --image foto.jpg

# Modo img + prompt → resposta
python scripts/infer.py \
  --checkpoint checkpoints/epoch_10.pt \
  --image foto.jpg \
  --prompt "What is shown in this image?"
```

## Parâmetros (22M)

| Submodelo | Params |
|---|---|
| ViT | 4.9M |
| Projector | 262K |
| Embedding + LM Head | 12.8M (tied) |
| Decoder (6×) | 3.9M |
| **Total** | **22.0M** |

Em fp32: ~88 MB. Com AdamW: ~700 MB. Folga enorme em qualquer GPU com ≥ 6GB.

## Licença

MIT
