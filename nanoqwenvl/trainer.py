import os
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_config: dict, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.config = train_config

        lr = train_config.get("learning_rate", 3e-4)
        wd = train_config.get("weight_decay", 0.1)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.clip = train_config.get("gradient_clip", 1.0)
        self.checkpoint_dir = train_config.get("checkpoint_dir", "checkpoints")
        self.num_prompt_tokens = train_config.get("num_prompt_tokens", 1)

        resume = train_config.get("resume")
        if resume:
            self.load(resume)

    def fit(self, train_loader, val_loader=None, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.get("num_epochs", 10)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            train_ppl = math.exp(train_loss)
            log = f"Epoch {epoch}/{num_epochs}  train loss={train_loss:.4f}  ppl={train_ppl:.2f}"

            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                val_ppl = math.exp(val_loss)
                log += f"  val loss={val_loss:.4f}  ppl={val_ppl:.2f}"

            print(log)

            ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
            self.save(ckpt_path, {"epoch": epoch, "train_loss": train_loss})

    def _train_epoch(self, loader):
        self.model.train()
        total = 0
        progress = tqdm(loader, desc="Train")
        for batch in progress:
            if len(batch) == 3:
                pixel_values, input_ids, prompt_lens = batch
            else:
                pixel_values, input_ids = batch
                prompt_lens = None
            pixel_values = pixel_values.to(self.device)
            input_ids = input_ids.to(self.device)
            logits = self.model(pixel_values, input_ids)
            loss = self._loss_fn(logits, input_ids, prompt_lens)

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            total += loss.item()
            progress.set_postfix(loss=loss.item())
        return total / len(loader)

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total = 0
        for batch in tqdm(loader, desc="Val"):
            if len(batch) == 3:
                pixel_values, input_ids, prompt_lens = batch
            else:
                pixel_values, input_ids = batch
                prompt_lens = None
            pixel_values = pixel_values.to(self.device)
            input_ids = input_ids.to(self.device)
            logits = self.model(pixel_values, input_ids)
            loss = self._loss_fn(logits, input_ids, prompt_lens)
            total += loss.item()
        return total / len(loader)

    def _loss_fn(self, logits, input_ids, prompt_lens=None):
        from .dataset import compute_loss
        pad_token_id = self.config.get("pad_token_id", 0)
        return compute_loss(
            logits, input_ids,
            self.model.num_visual_tokens,
            pad_token_id,
            prompt_lens=prompt_lens,
            num_prompt_tokens=self.num_prompt_tokens,
        )

    def save(self, path: str, extra: dict = None):
        state = {"model_state_dict": self.model.state_dict()}
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        print(f"Carregado checkpoint: {path}")
