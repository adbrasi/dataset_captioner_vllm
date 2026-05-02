"""Captioning de pares (Image1, Image2) usando o system prompt do dataset."""

import base64
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://adbrasi--gemma4-abliterated-serve.modal.run/v1"
MODEL = "gemma4-abliterated"

ROOT = Path(__file__).parent
SYSTEM_PROMPT = (ROOT / "system_prompt.txt").read_text()


def img_url(path: Path) -> dict:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def caption_pair(img1: Path, img2: Path, tags1: str = "", tags2: str = "") -> dict:
    user_text = (
        f"Image 1 booru tags: {tags1 or '(none)'}\n"
        f"Image 2 booru tags: {tags2 or '(none)'}\n\n"
        "Process the pair and output the JSON."
    )
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    img_url(img1),
                    img_url(img2),
                ],
            },
        ],
        max_tokens=1500,
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    latency = time.time() - t0
    return {
        "content": resp.choices[0].message.content,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "latency_s": latency,
    }


if __name__ == "__main__":
    img1 = Path(sys.argv[1])
    img2 = Path(sys.argv[2])
    tags1 = sys.argv[3] if len(sys.argv) > 3 else ""
    tags2 = sys.argv[4] if len(sys.argv) > 4 else ""

    r = caption_pair(img1, img2, tags1, tags2)
    print(r["content"])
    print(
        f"\n[prompt={r['prompt_tokens']}  completion={r['completion_tokens']}  "
        f"latency={r['latency_s']:.1f}s  tok/s={r['completion_tokens']/r['latency_s']:.1f}]"
    )
