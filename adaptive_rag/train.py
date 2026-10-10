"""Real CPU LoRA fine-tuning: contrastive positives and BM25 hard negatives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import functional as F

from .data import load_benchmark, save_split
from .encoder import Encoder, mean_pool
from .prepare import ENCODER_ID, ENCODER_REVISION
from .retrieval import BM25, top_indices


def contrastive_logits(query: torch.Tensor, document: torch.Tensor,
                       qids: list[str], dids: list[str], qrels: dict, temperature: float) -> torch.Tensor:
    scores = query @ document.T / temperature
    # Keep the designated positive; mask other known relevant/duplicate positives.
    mask = torch.tensor([[did in qrels[qid] and j != i for j, did in enumerate(dids)]
                         for i, qid in enumerate(qids)], device=scores.device)
    return scores.masked_fill(mask, -1e4)


def run(epochs: int = 4, seed: int = 42, batch_size: int = 8,
        learning_rate: float = 3e-4, max_length: int = 192) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()
    benchmark = load_benchmark()
    split = save_split(benchmark, Path("reports/split.json"), seed)
    docs = {d.id: d.content for d in benchmark.documents}
    ids = [d.id for d in benchmark.documents]
    sparse = BM25([docs[did] for did in ids])
    negatives = {}
    for qid in split["train"] + split["validation"]:
        order = top_indices(sparse.scores(benchmark.queries[qid]), 100)
        negatives[qid] = [ids[i] for i in order if ids[i] not in benchmark.train_qrels[qid]][:10]
        if not negatives[qid]:
            raise ValueError("Cannot mine a negative")
    encoder = Encoder(max_length=max_length)
    lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
                      lora_dropout=.05, target_modules=["query", "value"], bias="none")
    encoder.model = get_peft_model(encoder.model, lora)
    encoder.model.print_trainable_parameters()
    trainable = sum(p.numel() for p in encoder.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.model.parameters())
    optimizer = torch.optim.AdamW([p for p in encoder.model.parameters() if p.requires_grad],
                                 lr=learning_rate, weight_decay=.01)
    history = []
    output = Path("checkpoints/retriever_lora")
    output.mkdir(parents=True, exist_ok=True)

    def batch_loss(qids: list[str], training: bool, epoch: int = 0) -> torch.Tensor:
        positives = [sorted(benchmark.train_qrels[qid])[epoch % len(benchmark.train_qrels[qid])]
                     for qid in qids]
        hard = [negatives[qid][random.randrange(len(negatives[qid])) if training else 0]
                for qid in qids]
        document_ids = positives + hard
        q_features = encoder.tokenize([benchmark.queries[qid] for qid in qids])
        d_features = encoder.tokenize([docs[did] for did in document_ids])
        q_hidden = encoder.model(**q_features).last_hidden_state
        d_hidden = encoder.model(**d_features).last_hidden_state
        q_vectors = mean_pool(q_hidden, q_features["attention_mask"])
        d_vectors = mean_pool(d_hidden, d_features["attention_mask"])
        logits = contrastive_logits(q_vectors, d_vectors, qids, document_ids,
                                     benchmark.train_qrels, .05)
        return F.cross_entropy(logits, torch.arange(len(qids)))

    def validation_loss() -> float:
        encoder.model.eval()
        sums, count = 0., 0
        with torch.no_grad():
            for start in range(0, len(split["validation"]), batch_size):
                qids = split["validation"][start:start + batch_size]
                sums += float(batch_loss(qids, False)) * len(qids)
                count += len(qids)
        return sums / count

    best = validation_loss()
    encoder.model.save_pretrained(output, safe_serialization=True)
    history.append({"epoch": 0, "validation_contrastive_loss": best})
    best_epoch = 0
    print(f"Initial validation loss={best:.4f}; train={len(split['train'])}, val={len(split['validation'])}", flush=True)
    steps = 0
    for epoch in range(1, epochs + 1):
        encoder.model.train()
        order = split["train"].copy()
        random.shuffle(order)
        total_loss, count = 0., 0
        for start in range(0, len(order), batch_size):
            qids = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(qids, True, epoch - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in encoder.model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(qids)
            count += len(qids)
            steps += 1
            if steps % 20 == 0:
                print(f"epoch={epoch} step={steps} loss={total_loss/count:.4f}", flush=True)
        val = validation_loss()
        history.append({"epoch": epoch, "train_contrastive_loss": total_loss / count,
                        "validation_contrastive_loss": val})
        if val < best:
            best, best_epoch = val, epoch
            encoder.model.save_pretrained(output, safe_serialization=True)
        print(f"Epoch {epoch}: train={total_loss/count:.4f}; validation={val:.4f}; selected={best_epoch}", flush=True)
    # Record a public model ID instead of the author's absolute local path.
    config_path = output / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = ENCODER_ID
    config["revision"] = ENCODER_REVISION
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"seed": seed, "epochs_requested": epochs, "selected_epoch": best_epoch,
              "optimizer_steps": steps, "batch_size": batch_size, "max_length": max_length,
              "learning_rate": learning_rate, "temperature": .05, "negative_mining": "BM25 top-10, excludes all train-qrels positives",
              "validation_selection": "minimum held-out document-grouped contrastive loss; epoch 0 allowed",
              "trainable_parameters": trainable, "total_parameters": total,
              "trainable_fraction": trainable / total, "backend": "PyTorch CPU, float32, 4 threads",
              "seconds": time.perf_counter() - started, "base_model": ENCODER_ID,
              "base_revision": ENCODER_REVISION, "history": history}
    Path("reports/training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    run(args.epochs, args.seed, args.batch_size, max_length=args.max_length)
