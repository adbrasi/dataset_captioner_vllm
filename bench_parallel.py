"""Benchmark de throughput em batch — mede tok/s agregado em N requests paralelos."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle, islice
from pathlib import Path

from pair_caption import caption_pair

ROOT = Path(__file__).parent
IMG_A = ROOT / "1utuhrf72a_image_0001_A.jpg"
IMG_B = ROOT / "1utuhrf72a_image_0006_A.jpg"
IMG_C = ROOT / "g36yfwej12_image_0001_A.jpg"

BASE_PAIRS = [
    (IMG_A, IMG_B), (IMG_B, IMG_A), (IMG_A, IMG_C), (IMG_C, IMG_B),
    (IMG_B, IMG_C), (IMG_C, IMG_A), (IMG_A, IMG_B), (IMG_C, IMG_A),
]

CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N_PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else max(CONCURRENCY * 2, 8)

PAIRS = list(islice(cycle(BASE_PAIRS), N_PAIRS))

t0 = time.time()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    futs = [ex.submit(caption_pair, a, b) for a, b in PAIRS]
    results = [f.result() for f in as_completed(futs)]
wall = time.time() - t0

prompt_tot = sum(r["prompt_tokens"] for r in results)
comp_tot = sum(r["completion_tokens"] for r in results)
n = len(results)

print(f"=== concurrency={CONCURRENCY}, n={n} ===")
print(f"wall:             {wall:.1f}s")
print(f"prompt avg:       {prompt_tot/n:.0f}")
print(f"completion avg:   {comp_tot/n:.0f}")
print(f"agg tok/s:        {comp_tot/wall:.1f}")
print(f"per-pair (wall):  {wall/n:.2f}s")
print(f"$/100 pairs:      {wall/n * 100 * 0.001261:.3f}  (H200 @ $4.54/h)")
