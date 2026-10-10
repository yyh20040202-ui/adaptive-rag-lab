"""BEIR loading and document-grouped train/validation splits."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.title}\n{self.text}".strip()


@dataclass
class Benchmark:
    documents: list[Document]
    queries: dict[str, str]
    train_qrels: dict[str, dict[str, int]]
    test_qrels: dict[str, dict[str, int]]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                result[row["query-id"]][row["corpus-id"]] = score
    return dict(result)


def load_benchmark(root: str | Path = "data/raw/scifact") -> Benchmark:
    root = Path(root)
    docs = []
    with (root / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            docs.append(Document(str(row["_id"]), row.get("title", ""), row["text"]))
    queries = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    train, test = read_qrels(root / "qrels/train.tsv"), read_qrels(root / "qrels/test.tsv")
    if set(train) & set(test):
        raise ValueError("Official training and test query IDs overlap")
    if len({doc.id for doc in docs}) != len(docs):
        raise ValueError("Duplicate corpus IDs")
    valid = {doc.id for doc in docs}
    for qrels in (train, test):
        if not set(qrels) <= queries.keys():
            raise ValueError("Missing query text")
        if any(not set(rels) <= valid for rels in qrels.values()):
            raise ValueError("Missing relevant document")
    return Benchmark(docs, queries, train, test)


def grouped_split(qrels: dict[str, dict[str, int]], seed: int = 42,
                  validation_fraction: float = .2) -> tuple[list[str], list[str]]:
    """Claims citing the same paper stay in one partition, transitively.

    The retrieval corpus is intentionally shared; supervision is separated.
    No official test labels enter this function.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    parent = {qid: qid for qid in qrels}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for qid, rels in sorted(qrels.items()):
        for did in rels:
            if did in owner:
                parent[find(qid)] = find(owner[did])
            else:
                owner[did] = qid
    groups = defaultdict(list)
    for qid in sorted(qrels):
        groups[find(qid)].append(qid)
    components = list(groups.values())
    random.Random(seed).shuffle(components)
    validation, training = [], []
    target = round(len(qrels) * validation_fraction)
    for group in components:
        (validation if len(validation) < target else training).extend(group)
    if not training or not validation:
        raise ValueError("Not enough independent document groups to split")
    train_docs = {did for qid in training for did in qrels[qid]}
    val_docs = {did for qid in validation for did in qrels[qid]}
    assert not train_docs & val_docs
    return sorted(training), sorted(validation)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_split(benchmark: Benchmark, output: Path, seed: int = 42) -> dict:
    train, val = grouped_split(benchmark.train_qrels, seed)
    result = {"seed": seed, "train": train, "validation": val,
              "test": sorted(benchmark.test_qrels), "corpus_documents": len(benchmark.documents),
              "protocol": "BEIR SciFact test; document-grouped split of official train labels"}
    root = Path("data/raw/scifact")
    result["input_sha256"] = {str(p.relative_to(root)): file_sha256(p) for p in
                             [root / "corpus.jsonl", root / "queries.jsonl",
                              root / "qrels/train.tsv", root / "qrels/test.tsv"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
