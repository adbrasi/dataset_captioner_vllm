"""
Cliente OpenAI-compatible para o serve do Modal.

Uso:
    .venv/bin/python client_query.py                                # imagem default
    .venv/bin/python client_query.py path/to/outra_imagem.jpg
    .venv/bin/python client_query.py path/to/img.png "prompt customizado"
"""

import base64
import sys
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://adbrasi--gemma4-abliterated-serve.modal.run/v1"
MODEL = "gemma4-abliterated"
DEFAULT_IMAGE = Path(__file__).parent / "g36yfwej12_image_0001_A.jpg"
DEFAULT_PROMPT = (
    "Descreva esta imagem em detalhes (em pt-BR): o que aparece, "
    "composição, cores, estilo visual, e qualquer texto visível."
)

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT

mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
b64 = base64.b64encode(image_path.read_bytes()).decode()

client = OpenAI(base_url=BASE_URL, api_key="not-needed")

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ],
    max_tokens=800,
    temperature=0.4,
)

print(resp.choices[0].message.content)
print(f"\n[tokens: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens}]")
