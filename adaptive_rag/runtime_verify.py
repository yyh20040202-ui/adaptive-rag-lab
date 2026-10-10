"""Small serving-consistency check after changing only runtime cache settings."""
import json
from pathlib import Path
import subprocess
from .data import load_benchmark
from .engine import RetrievalEngine
from .resources import snapshot


def observe():
    import psutil
    owned = []
    for process in psutil.process_iter(['name', 'cmdline', 'memory_info']):
        try:
            info = process.info
            if (info['name'] or '').lower() == 'llama-server.exe' and any('adaptive-rag-lab' in x for x in info['cmdline'] or []):
                args = info['cmdline'] or []
                owned.append({'port': args[args.index('--port')+1], 'rss_gib': info['memory_info'].rss/1024**3,
                              'cache_ram': args[args.index('--cache-ram')+1],
                              'ctx_checkpoints': args[args.index('--ctx-checkpoints')+1],
                              'gpu_layers': args[args.index('--n-gpu-layers')+1],
                              'load_mode': args[args.index('--load-mode')+1] if '--load-mode' in args else 'auto',
                              'context': args[args.index('--ctx-size')+1]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    used, total, temperature = map(int, gpu.stdout.strip().split(','))
    return {'system': snapshot(), 'owned_llama_processes': owned,
            'gpu_memory_used_mib': used, 'gpu_memory_total_mib': total, 'gpu_temperature_c': temperature}


if __name__ == '__main__':
    data = load_benchmark()
    rows = [json.loads(line) for line in Path('reports/quality_per_query.jsonl').read_text().splitlines()]
    engine = RetrievalEngine.from_corpus('scifact')
    results, samples = [], [observe()]
    for index in [0, 100, 150, 200]:
        row = rows[index]
        answer = engine.search(data.queries[row['query_id']], 'cross_encoder', top_k=10)
        ranked = [hit['source_id'] for hit in answer['hits']]
        results.append({'query_id': row['query_id'], 'expected': row['top10']['cross_encoder'],
                        'actual': ranked, 'match': ranked == row['top10']['cross_encoder']})
        samples.append(observe())
    report = {'passed': all(row['match'] for row in results), 'checked_queries': len(results),
              'rankings': results, 'resources': samples,
              'scope': 'Same fixed queries used previously. Checks serving rank identity after cache settings; not a new 300-query benchmark or QA evaluation. Memory values are sampled, not continuous peaks.'}
    Path('reports/runtime_verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checked_queries': len(results), 'last_resources': samples[-1]}, indent=2))
    if not report['passed']:
        raise SystemExit(1)
