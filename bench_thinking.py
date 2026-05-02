"""Mede throughput agregado de captioning REAL (com booru cache + thinking) em N pares.

Reusa a mesma pipeline do batch_caption.py. Roda N requests simultâneos
contra o backend escolhido, mede tokens/s, $/100, ETA pra 3500.

Uso:
    .venv/bin/python bench_thinking.py /tmp/test_batch --backend qwen --concurrency 16 --pairs 16
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle, islice
from pathlib import Path

from openai import OpenAI

from batch_caption import (
    BACKENDS, SYSTEM_PROMPT_FILE, discover_pairs, load_booru_cache,
    populate_booru_cache, caption_one,
)

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("--backend", choices=list(BACKENDS), required=True)
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--pairs", type=int, default=32, help="quantos pares rodar (cycla a pasta)")
args = ap.parse_args()

cfg = BACKENDS[args.backend]
extra_body = {}
if cfg["supports_thinking"]:
    extra_body["chat_template_kwargs"] = {"enable_thinking": True}

system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
client = OpenAI(base_url=cfg["base_url"], api_key="not-needed", max_retries=0)

pairs_avail = discover_pairs(args.folder)
booru = load_booru_cache(args.folder)
populate_booru_cache(args.folder, pairs_avail, booru)

work = list(islice(cycle(pairs_avail), args.pairs))

t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
    futs = [
        ex.submit(caption_one, client, cfg["model"], system_prompt, p,
                  booru[p.key]["a"], booru[p.key]["b"], extra_body)
        for p in work
    ]
    for f in as_completed(futs):
        try:
            results.append(f.result())
        except Exception as exc:
            print("ERR:", exc, file=sys.stderr)

wall = time.time() - t0
n = len(results)
comp = sum(r["completion_tokens"] for r in results)
H200_PER_SEC = 4.54 / 3600

print(f"=== backend={args.backend} concurrency={args.concurrency} n={n}/{args.pairs} ===")
print(f"wall:           {wall:.1f}s")
print(f"completion tot: {comp}  avg={comp/max(n,1):.0f}")
print(f"agg tok/s:      {comp/wall:.1f}")
print(f"per-pair (wall):{wall/max(n,1):.2f}s")
print(f"$/100 pairs:    {wall/max(n,1) * 100 * H200_PER_SEC:.3f}")
print(f"ETA 3500 pares: {wall/max(n,1) * 3500 / 60:.1f} min  custo: ${wall/max(n,1) * 3500 * H200_PER_SEC:.2f}")
