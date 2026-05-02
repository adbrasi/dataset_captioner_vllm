"""Captioning de vídeo NATIVO com Gemma 4 abliterated via vLLM.

Envia o vídeo inteiro como `video_url` (data-URI base64). O Gemma 4
internamente extrai até 32 frames e insere timestamps mm:ss entre eles
pra reasoning temporal.

Uso:
    .venv/bin/python video_caption.py /caminho/video.mp4
    .venv/bin/python video_caption.py /caminho/video.mp4 -o caption.txt
    .venv/bin/python video_caption.py /caminho/video.mp4 --compress  # pré-comprime se for grande
"""

import argparse
import base64
import logging
import mimetypes
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://adbrasi--gemma4-abliterated-serve.modal.run/v1"
MODEL = "gemma4-abliterated"
SYSTEM_PROMPT_FILE = Path(__file__).parent / "video_system_prompt_test.txt"

# Acima desse tamanho (após base64 = ~1.33x), é arriscado mandar via HTTP.
# Recomendado pré-comprimir.
WARN_BYTES = 25 * 1024 * 1024
HARD_BYTES = 80 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("video")


def compress_video(src: Path, dst: Path, max_dim: int = 720, fps: int = 8) -> None:
    """Re-encoda em H.264 baixinho — mantém qualidade ok pro Gemma 4 e
    diminui drasticamente o payload."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", f"scale='min({max_dim},iw)':-2:flags=lanczos,fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-an",  # remove áudio (Gemma 4 31B dense não usa áudio)
        "-movflags", "+faststart",
        str(dst),
    ]
    log.info("comprimindo: %s", " ".join(cmd[5:]))
    subprocess.run(cmd, check=True)


def video_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "video/mp4"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def caption_video(video: Path, max_tokens: int) -> str:
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    client = OpenAI(base_url=BASE_URL, api_key="not-needed", max_retries=2)

    size = video.stat().st_size
    log.info("vídeo: %s  %.2f MB", video.name, size / 1024 / 1024)
    if size > HARD_BYTES:
        raise RuntimeError(
            f"vídeo muito grande ({size/1024/1024:.1f} MB > {HARD_BYTES/1024/1024:.0f}). "
            "Use --compress."
        )
    if size > WARN_BYTES:
        log.warning("vídeo grande — payload base64 vai ficar ~%.0f MB. Considere --compress.",
                    size * 4 / 3 / 1024 / 1024)

    data_uri = video_data_uri(video)
    log.info("data URI montada (%.1f MB base64). Enviando...", len(data_uri) / 1024 / 1024)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_uri}},
                    {"type": "text", "text": "Describe the video following the system instructions exactly."},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=0.5,
        timeout=900,
    )
    elapsed = time.time() - t0
    u = resp.usage
    log.info(
        "ok em %.1fs  prompt=%d  completion=%d  finish=%s",
        elapsed, u.prompt_tokens, u.completion_tokens, resp.choices[0].finish_reason,
    )
    return resp.choices[0].message.content


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--compress", action="store_true",
                    help="pré-comprime com ffmpeg antes de mandar (recomendado >25MB)")
    ap.add_argument("--compress-dim", type=int, default=720,
                    help="lado maior alvo no compress (default 720)")
    ap.add_argument("--compress-fps", type=int, default=8,
                    help="fps alvo no compress (default 8 — Gemma 4 amostra ~32 frames internamente)")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.video.exists():
        log.error("vídeo não existe: %s", args.video)
        return 1

    output = args.output or args.video.with_suffix(".txt")

    if args.compress:
        with tempfile.TemporaryDirectory(prefix="vidcomp_") as tmp:
            compressed = Path(tmp) / (args.video.stem + "_compressed.mp4")
            compress_video(args.video, compressed, args.compress_dim, args.compress_fps)
            log.info("comprimido: %.2f MB → %.2f MB",
                     args.video.stat().st_size / 1024 / 1024,
                     compressed.stat().st_size / 1024 / 1024)
            caption = caption_video(compressed, args.max_tokens)
    else:
        caption = caption_video(args.video, args.max_tokens)

    output.write_text(caption.strip() + "\n", encoding="utf-8")
    log.info("salvo em %s", output)
    print()
    print(caption.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
