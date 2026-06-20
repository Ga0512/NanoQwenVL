import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from collections import defaultdict
from abc import ABC, abstractmethod


class ImageCaptionDataset(Dataset, ABC):
    @abstractmethod
    def __getitem__(self, idx):
        ...


class CsvCaptionDataset(ImageCaptionDataset):
    def __init__(self, csv_path, transform, tokenizer, max_length=64,
                 prompt_template=None, sep="\t"):
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Tuple[str, str]] = []
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))

        self.prompt_template = prompt_template
        if prompt_template:
            self.prompt_ids = tokenizer.encode(prompt_template)
            self.num_prompt_tokens = 1 + len(self.prompt_ids)
        else:
            self.num_prompt_tokens = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        from PIL import Image
        image = self.transform(Image.open(img_path).convert("RGB"))
        return image, self._tokenize(caption)

    def _tokenize(self, caption):
        cap_ids = self.tokenizer.encode(caption, max_length=self.max_length - 2,
                                        truncation=True)
        if self.prompt_template:
            ids = [self.tokenizer.bos_token_id] + self.prompt_ids + cap_ids + [self.tokenizer.eos_token_id]
        else:
            ids = [self.tokenizer.bos_token_id] + cap_ids + [self.tokenizer.eos_token_id]
        return torch.tensor(ids, dtype=torch.long)


FLICKR8K_CAPTION_KEYS = ["caption_0", "caption_1", "caption_2", "caption_3", "caption_4"]


class Flickr8kHF(ImageCaptionDataset):
    """Carrega Flickr8k do HuggingFace (jxie/flickr8k) com carregamento lazy.

    5 legendas por imagem. Imagens são baixadas sob demanda (não na __init__).
    """
    def __init__(self, split, transform, tokenizer, max_length=64,
                 max_samples=None, prompt_template=None, synthetic=False):
        from datasets import load_dataset

        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.synthetic = synthetic
        self.prompt_template = prompt_template

        if prompt_template:
            self.prompt_ids = tokenizer.encode(prompt_template)
            self.num_prompt_tokens = 1 + len(self.prompt_ids)
        else:
            self.num_prompt_tokens = 1

        self._ds = load_dataset("jxie/flickr8k", split=split)
        self._indices: List[Tuple[int, str]] = []
        for i in range(len(self._ds)):
            for k in FLICKR8K_CAPTION_KEYS:
                self._indices.append((i, k))

        if max_samples is not None:
            self._indices = self._indices[:max_samples]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, key = self._indices[idx]
        item = self._ds[i]
        if self.synthetic:
            image = torch.randn(3, 224, 224)
        else:
            image = self.transform(item["image"].convert("RGB"))
        return image, self._tokenize(item[key])

    def _tokenize(self, caption):
        cap_ids = self.tokenizer.encode(caption, max_length=self.max_length - 2,
                                        truncation=True)
        if self.prompt_template:
            ids = [self.tokenizer.bos_token_id] + self.prompt_ids + cap_ids + [self.tokenizer.eos_token_id]
        else:
            ids = [self.tokenizer.bos_token_id] + cap_ids + [self.tokenizer.eos_token_id]
        return torch.tensor(ids, dtype=torch.long)


class Flickr8kLocal(ImageCaptionDataset):
    ANNOTATIONS_DIR = "/tmp/flickr8k/text"

    def __init__(self, split, transform, tokenizer, max_length=64,
                 image_dir=None, synthetic=False, max_samples=None,
                 prompt_template=None):
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_dir = image_dir
        self.synthetic = synthetic
        self.prompt_template = prompt_template

        if prompt_template:
            self.prompt_ids = tokenizer.encode(prompt_template)
            self.num_prompt_tokens = 1 + len(self.prompt_ids)
        else:
            self.num_prompt_tokens = 1

        caps = defaultdict(list)
        tok_path = os.path.join(self.ANNOTATIONS_DIR, "Flickr8k.token.txt")
        with open(tok_path) as f:
            for line in f:
                img, caption = line.strip().split("\t")
                caps[img.split("#")[0]].append(caption)

        split_file = {
            "train": "Flickr_8k.trainImages.txt",
            "validation": "Flickr_8k.devImages.txt",
            "test": "Flickr_8k.testImages.txt",
        }.get(split, "Flickr_8k.trainImages.txt")

        with open(os.path.join(self.ANNOTATIONS_DIR, split_file)) as f:
            split_imgs = {line.strip() for line in f}

        self.samples: List[Tuple[str, str]] = []
        for name in sorted(split_imgs):
            if name in caps:
                for caption in caps[name]:
                    self.samples.append((name, caption))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        if self.synthetic:
            image = torch.randn(3, 224, 224)
        else:
            from PIL import Image
            path = os.path.join(self.image_dir, img_name)
            image = self.transform(Image.open(path).convert("RGB"))
        return image, self._tokenize(caption)

    def _tokenize(self, caption):
        cap_ids = self.tokenizer.encode(caption, max_length=self.max_length - 2,
                                        truncation=True)
        if self.prompt_template:
            ids = [self.tokenizer.bos_token_id] + self.prompt_ids + cap_ids + [self.tokenizer.eos_token_id]
        else:
            ids = [self.tokenizer.bos_token_id] + cap_ids + [self.tokenizer.eos_token_id]
        return torch.tensor(ids, dtype=torch.long)


DRIVELM_CAMERA_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]


class DriveLMNuScenesDataset(ImageCaptionDataset):
    """DriveLM-nuScenes via ac4462/DriveLM-reasoning + imagens nuScenes.

    Usa só CAM_FRONT (1600×900 → resize 224×224).
    Modo: (imagem, pergunta) → resposta.

    O dataset `ac4462/DriveLM-reasoning` tem 5280 QA pairs (train-only).
    Imagens: precisa do diretório nuScenes em nuscenes_root ou synthetic=True.
    """
    def __init__(self, split, transform, tokenizer, max_length=128,
                 max_samples=None, prompt_template=None, synthetic=False,
                 nuscenes_root=None):
        from datasets import load_dataset

        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.synthetic = synthetic
        self.nuscenes_root = nuscenes_root

        if prompt_template:
            self.prompt_ids = tokenizer.encode(prompt_template)
            self.num_prompt_tokens = 1 + len(self.prompt_ids)
        else:
            self.prompt_ids = []
            self.num_prompt_tokens = 1

        ds = load_dataset("ac4462/DriveLM-reasoning", split="train")
        self.samples: List[Tuple[str, str, List[str]]] = []
        for item in ds:
            q = item["problem"]
            a = item["solution"]
            paths = item["image"]
            self.samples.append((q, a, paths))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, answer, img_paths = self.samples[idx]

        if self.synthetic:
            image = torch.randn(3, 224, 224)
        else:
            image = self._load_camera_grid(img_paths)

        token_ids, prompt_len = self._tokenize(question, answer)
        return image, token_ids, prompt_len

    def _load_camera_grid(self, img_paths):
        from PIL import Image
        # Usa só CAM_FRONT
        cam_path = [p for p in img_paths if "CAM_FRONT" in p and "CAM_FRONT_LEFT" not in p and "CAM_FRONT_RIGHT" not in p]
        p = cam_path[0] if cam_path else img_paths[0]
        resolved = self._resolve_path(p)
        return self.transform(Image.open(resolved).convert("RGB"))

    def _resolve_path(self, path):
        if self.nuscenes_root is not None:
            name = os.path.basename(path)
            cam = [c for c in DRIVELM_CAMERA_ORDER if c in path]
            cam = cam[0] if cam else "CAM_FRONT"
            return os.path.join(self.nuscenes_root, cam, name)
        return os.path.abspath(path)

    def _tokenize(self, question, answer):
        q_ids = self.tokenizer.encode(question, max_length=self.max_length - 3,
                                      truncation=True)
        a_ids = self.tokenizer.encode(answer, max_length=self.max_length - 2 - len(q_ids),
                                      truncation=True)
        prompt_ids = self.prompt_ids + q_ids
        prompt_len = 1 + len(prompt_ids)
        ids = [self.tokenizer.bos_token_id] + prompt_ids + a_ids + [self.tokenizer.eos_token_id]
        return torch.tensor(ids, dtype=torch.long), prompt_len


def collate_fn(batch, pad_token_id: int):
    """batch: list of (image, token_ids) or (image, token_ids, prompt_len)."""
    if len(batch[0]) == 3:
        images, texts, prompt_lens = zip(*batch)
        prompt_lens = torch.tensor(prompt_lens, dtype=torch.long)
    else:
        images, texts = zip(*batch)
        prompt_lens = None
    images = torch.stack(images, dim=0)
    max_len = max(t.size(0) for t in texts)
    padded = torch.full((len(texts), max_len), pad_token_id, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i, : t.size(0)] = t
    if prompt_lens is not None:
        return images, padded, prompt_lens
    return images, padded


def compute_loss(logits, input_ids, num_visual_tokens, pad_token_id,
                 prompt_lens=None, num_prompt_tokens=1):
    B, T = input_ids.shape
    seq_len = logits.size(1)

    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100

    if prompt_lens is not None:
        # Per-sample prompt lengths (for dynamic prompts, e.g. DriveLM)
        prompt_lens = prompt_lens.to(logits.device)
        targets = torch.full((B, seq_len), -100, device=logits.device, dtype=torch.long)
        for b in range(B):
            npt = int(prompt_lens[b].item())
            num_resp = T - npt
            if num_resp > 0:
                rs = num_visual_tokens + npt
                targets[b, rs : rs + num_resp] = labels[b, npt:]
    else:
        # Fixed prompt length (Flickr8k, CSV)
        num_response_tokens = T - num_prompt_tokens
        targets = torch.full((B, seq_len), -100, device=logits.device, dtype=torch.long)
        if num_response_tokens > 0:
            resp_start = num_visual_tokens + num_prompt_tokens
            targets[:, resp_start : resp_start + num_response_tokens] = labels[:, num_prompt_tokens:]

    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        targets[:, 1:].reshape(-1),
        ignore_index=-100,
    )
