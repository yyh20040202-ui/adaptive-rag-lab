"""Joint query-document token attention, bounded CPU inference and shared blending."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .retrieval import normalized_scores, top_indices


def blend_order(candidates, logits, fusion_scores, alpha: float, depth: int):
    candidates = np.asarray(candidates[:depth], dtype=int)
    values = alpha * normalized_scores(np.asarray(logits)[:len(candidates)]) + (1-alpha) * normalized_scores(np.asarray(fusion_scores)[candidates])
    return candidates[top_indices(values, len(candidates))], values


class CrossEncoder:
    def __init__(self, model_path="models/cross_encoder_l6", max_length=384, batch_size=8, quantized=True):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from .resources import CPU_THREADS, check_resources, ResourceRecorder
        check_resources()
        self.recorder = ResourceRecorder()
        self.recorder.sample()
        torch.set_num_threads(CPU_THREADS)
        self.max_length = max_length
        self.batch_size = max(1, min(batch_size, 8))
        self.path = Path(model_path)
        self.provenance = json.loads((self.path / "provenance.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.path), local_files_only=True,
                     trust_remote_code=False, attn_implementation="sdpa", use_safetensors=True).eval()
        self.quantized = quantized
        if quantized:
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.recorder.sample()

    def predict(self, query: str, texts: list[str]) -> np.ndarray:
        import torch
        # Preserve full bounded queries; do not silently change a user's question.
        if len(self.tokenizer.encode(query, add_special_tokens=False)) > 128:
            raise ValueError("质量模式的问题请控制在128个模型token以内。")
        values = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                self.recorder.sample()
                batch = texts[start:start+self.batch_size]
                # Only document tokens are truncated: the complete bounded query is kept.
                inputs = self.tokenizer([query]*len(batch), batch, padding=True, truncation="only_second",
                                        max_length=self.max_length, return_tensors="pt")
                values.extend(self.model(**inputs).logits.flatten().cpu().tolist())
        return np.asarray(values, dtype=np.float32)
