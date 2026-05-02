"""Batch captioning de pares (_A, _B) numa pasta. Multi-backend (Gemma 4 ou Qwen 3.6).

Para cada par <prefix>_image_NNNN_A.<ext> + <prefix>_image_NNNN_B.<ext>:
  - lê booru tags de <name>_A.txt e <name>_B.txt
  - manda IMG_A como Image 1 e IMG_B como Image 2 pro modelo
  - parseia JSON da resposta, extrai "short_prompt"
  - sobrescreve <name>_B.txt com o short_prompt

Idempotente: registra cada par concluído em .processed-<backend>.jsonl. Logs de processed
SÃO POR BACKEND — re-rodar com outro backend reprocessa tudo (intencional).

Uso:
    .venv/bin/python batch_caption.py /caminho/da/pasta --backend gemma
    .venv/bin/python batch_caption.py /caminho/da/pasta --backend qwen
    .venv/bin/python batch_caption.py /caminho/da/pasta --backend qwen --no-thinking
    .venv/bin/python batch_caption.py /caminho/da/pasta --backend gemma --max-pairs 10
"""

import argparse
import base64
import io
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from openai import OpenAI
from PIL import Image, ImageOps

BACKENDS: dict[str, dict] = {
    "gemma": {
        "base_url": "https://adbrasi--gemma4-abliterated-serve.modal.run/v1",
        "model": "gemma4-abliterated",
        "supports_thinking": False,
    },
    "qwen": {
        "base_url": "https://adbrasi--qwen36-abliterated-serve.modal.run/v1",
        "model": "qwen36-abliterated",
        "supports_thinking": True,
    },
}

SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_RETRIES = 3
REQUEST_TIMEOUT = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch")


@dataclass(frozen=True)
class Pair:
    img_a: Path
    img_b: Path
    txt_a: Path
    txt_b: Path

    @property
    def key(self) -> str:
        return self.img_b.stem.removesuffix("_B")


def discover_pairs(folder: Path) -> list[Pair]:
    pairs: list[Pair] = []
    a_images = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.stem.endswith("_A")]
    a_images.sort()

    for a in a_images:
        base = a.stem[:-2]
        b = None
        for ext in (a.suffix, *IMG_EXTS):
            cand = folder / f"{base}_B{ext}"
            if cand.exists():
                b = cand
                break
        if b is None:
            log.warning("sem par _B para %s — skip", a.name)
            continue
        pairs.append(Pair(
            img_a=a,
            img_b=b,
            txt_a=a.with_suffix(".txt"),
            txt_b=b.with_suffix(".txt"),
        ))
    return pairs


def load_processed(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    keys = set()
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            keys.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


def append_processed(log_path: Path, key: str, lock: Lock) -> None:
    line = json.dumps({"key": key, "ts": time.time()}, ensure_ascii=False) + "\n"
    with lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


def img_part(path: Path) -> dict:
    """Carrega imagem aplicando EXIF transpose (PIL não roda auto), converte
    pra RGB e re-encoda em JPEG base64. Garante que o vLLM recebe pixels na
    orientação correta — sem isso, JPEGs com EXIF Orientation chegam tortos."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def read_tags(p: Path) -> str:
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace").strip()


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_short_prompt(raw: str) -> str:
    """Parseia o JSON da resposta e devolve short_prompt.

    Tolera ```json ... ``` mesmo o system prompt pedindo JSON puro.
    """
    text = raw.strip()
    try:
        return json.loads(text)["short_prompt"]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))["short_prompt"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    raise ValueError(f"short_prompt não encontrado em: {text[:300]}...")


def build_messages(system_prompt: str, pair: Pair, tags_a: str, tags_b: str) -> list[dict]:
    user_text = (
        f"tags booru of image 1: {tags_a or '(none)'}\n\n"
        f"tags booru of image 2: {tags_b or '(none)'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                img_part(pair.img_a),
                img_part(pair.img_b),
                {"type": "text", "text": user_text},
            ],
        },
    ]


def caption_one(client: OpenAI, model: str, system_prompt: str, pair: Pair, extra_body: dict | None) -> dict:
    tags_a = read_tags(pair.txt_a)
    tags_b = read_tags(pair.txt_b)
    messages = build_messages(system_prompt, pair, tags_a, tags_b)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1500,
                temperature=0.4,
                response_format={"type": "json_object"},
                timeout=REQUEST_TIMEOUT,
                extra_body=extra_body or {},
            )
            short_prompt = extract_short_prompt(resp.choices[0].message.content)
            return {
                "short_prompt": short_prompt,
                "latency": time.time() - t0,
                "completion_tokens": resp.usage.completion_tokens,
            }
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            log.warning("[%s] tentativa %d falhou (%s) — esperando %ds", pair.key, attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"falha após {MAX_RETRIES} tentativas no par {pair.key}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="pasta com imagens _A e _B")
    ap.add_argument("--backend", choices=list(BACKENDS), required=True,
                    help="qual modelo usar (gemma | qwen)")
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--max-pairs", type=int, default=None, help="limite p/ teste")
    ap.add_argument("--processed-log", type=Path, default=None,
                    help="caminho do log de progresso (default: <folder>/.processed-<backend>.jsonl)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="desativa thinking no Qwen (default ON; ignorado no Gemma)")
    args = ap.parse_args()

    if not args.folder.is_dir():
        log.error("não é uma pasta: %s", args.folder)
        return 1

    cfg = BACKENDS[args.backend]
    log.info("backend=%s  model=%s  url=%s", args.backend, cfg["model"], cfg["base_url"])

    extra_body: dict = {}
    if cfg["supports_thinking"]:
        thinking = not args.no_thinking
        if not thinking:
            log.warning(
                "thinking=False + response_format=json_object pode quebrar (vLLM #18819). "
                "Mantendo response_format mesmo assim — se quebrar, retire --no-thinking."
            )
        extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}
        log.info("thinking=%s", thinking)

    log.info("descobrindo pares em %s", args.folder)
    pairs = discover_pairs(args.folder)
    log.info("encontrados %d pares", len(pairs))

    processed_log = args.processed_log or (args.folder / f".processed-{args.backend}.jsonl")
    done_keys = load_processed(processed_log)
    pending = [p for p in pairs if p.key not in done_keys]
    log.info("já processados: %d  |  pendentes: %d", len(done_keys), len(pending))

    if args.max_pairs is not None:
        pending = pending[: args.max_pairs]
        log.info("limitando a %d pares (--max-pairs)", len(pending))

    if not pending:
        log.info("nada a fazer.")
        return 0

    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    client = OpenAI(base_url=cfg["base_url"], api_key="not-needed", max_retries=0)
    write_lock = Lock()

    ok = 0
    fail = 0
    t_start = time.time()
    total_completion = 0

    def worker(pair: Pair) -> tuple[Pair, str | None, str | None, dict | None]:
        try:
            r = caption_one(client, cfg["model"], system_prompt, pair, extra_body)
            atomic_write(pair.txt_b, r["short_prompt"])
            append_processed(processed_log, pair.key, write_lock)
            return pair, None, r["short_prompt"], r
        except Exception as exc:
            return pair, str(exc), None, None

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(worker, p): p for p in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            pair, err, _short, meta = fut.result()
            if err is None:
                ok += 1
                total_completion += meta["completion_tokens"]
                if i % 10 == 0 or i == len(pending):
                    elapsed = time.time() - t_start
                    rate = i / elapsed
                    eta = (len(pending) - i) / rate if rate else 0
                    log.info(
                        "%d/%d ok=%d fail=%d  rate=%.2f pair/s  eta=%.0fs  comp_tokens_avg=%.0f",
                        i, len(pending), ok, fail, rate, eta, total_completion / max(ok, 1),
                    )
            else:
                fail += 1
                log.error("[%s] FAIL: %s", pair.key, err)

    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info("done: ok=%d fail=%d em %.1fs  (%.2f pair/s)", ok, fail, elapsed, len(pending) / elapsed)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
