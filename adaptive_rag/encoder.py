"""Local-only MiniLM mean pooling and optional PEFT adapter inference."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from .resources import CPU_THREADS, check_resources


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return torch.nn.functional.normalize(pooled, p=2, dim=-1)


class Encoder:
    def __init__(self, model_path: str | Path = "models/minilm", adapter: str | Path | None = None,
                 max_length: int = 192, threads: int = CPU_THREADS):
        check_resources()
        torch.set_num_threads(threads)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True,
                                               trust_remote_code=False, attn_implementation="sdpa")
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter), local_files_only=True)
        self.model.eval()

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                              return_tensors="pt")

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 8, progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        batch_size = max(1, min(batch_size, 8))
        batches = []
        for start in range(0, len(texts), batch_size):
            check_resources()
            features = self.tokenize(texts[start:start + batch_size])
            hidden = self.model(**features).last_hidden_state
            batches.append(mean_pool(hidden, features["attention_mask"]).cpu().numpy())
            if progress and (start // batch_size) % 20 == 0:
                print(f"Encoded {min(start+batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.concatenate(batches).astype(np.float32)
