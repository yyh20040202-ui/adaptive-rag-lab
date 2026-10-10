"""Hash current public artifacts and pinned local inputs after final evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from .data import file_sha256


def run():
    root = Path('.')
    tracked = []
    for folder in ['adaptive_rag', 'tests', 'docs', 'reports', 'checkpoints', '.github']:
        tracked.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                       and p.suffix in {'.py', '.html', '.css', '.js', '.md', '.json', '.jsonl', '.png', '.svg', '.safetensors', '.yml', '.yaml'})
    tracked.extend(root/name for name in ['README.md', 'MODEL_CARD.md', 'DATA_CARD.md', 'LICENSE', 'pyproject.toml', 'requirements.txt', 'requirements-demo.txt', '.gitignore'] if (root/name).exists())
    manifest_path = root/'reports/artifact_manifest.json'
    files = {p.as_posix(): {'sha256': file_sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(set(tracked)) if p != manifest_path}
    inputs = {name: file_sha256(root/'data/raw/scifact'/name)
              for name in ['corpus.jsonl', 'queries.jsonl', 'qrels/train.tsv', 'qrels/test.tsv']}
    report = {'schema': 2, 'description': 'Current public source, checkpoint and report hashes; excludes this manifest to avoid a circular hash. Historical V1 manifest retained separately.',
              'files': files, 'input_sha256': inputs,
              'provenance_report': 'reports/runtime_environment.json',
              'retrieval_config': 'checkpoints/quality_config.json',
              'scope': 'Input and source identity, not a proof of model quality or author independence.'}
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'hashed_public_files': len(files), 'hashed_inputs': len(inputs)}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
