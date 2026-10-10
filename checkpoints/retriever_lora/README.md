---
base_model: sentence-transformers/all-MiniLM-L6-v2
library_name: peft
license: apache-2.0
tags: [lora, retrieval, scifact]
---

# SciFact retrieval LoRA

Actual CPU-trained MiniLM query/value adapter: r=8, alpha=16, dropout=0.05, 73,728 trainable parameters. Train/validation 646/163 document-grouped BEIR SciFact claims; checkpoint selected at epoch 1 by validation contrastive loss from a four-epoch run. Maximum input length 192 tokens, mean pooling and L2 normalization.

Load the pinned local MiniLM base first, then call `PeftModel.from_pretrained(base, "checkpoints/retriever_lora", local_files_only=True)`. The repository contains delta weights only. Base attribution, licenses, evaluation and limitations are in the root MODEL_CARD.md and reports/.
