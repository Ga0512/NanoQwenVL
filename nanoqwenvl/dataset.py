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


def collate_fn(batch, pad_token_id: int):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    max_len = max(t.size(0) for t in texts)
    padded = torch.full((len(texts), max_len), pad_token_id, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i, : t.size(0)] = t
    return images, padded


def compute_loss(logits, input_ids, num_visual_tokens, pad_token_id,
                 num_prompt_tokens=1):
    B, T = input_ids.shape
    seq_len = logits.size(1)

    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100

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
