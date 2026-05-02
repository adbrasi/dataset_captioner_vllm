import base64
import json
import os
import sys
import urllib.request

API_KEY = os.environ["FEATHERLESS_API_KEY"]
BASE_URL = "https://api.featherless.ai/v1"
MODEL = "huihui-ai/gemma-3-27b-it-abliterated"
IMAGE_PATH = "/home/adolfocesar/projects/runpod_llm/g36yfwej12_image_0001_A.jpg"

with open(IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Descreva esta imagem em detalhes (em pt-BR): o que aparece, composição, cores, estilo, e qualquer texto visível. Seja minucioso."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }
    ],
    "max_tokens": 800,
    "temperature": 0.4,
}

req = urllib.request.Request(
    f"{BASE_URL}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "featherless-test/1.0",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
    sys.exit(1)

print("=" * 70)
print(f"Modelo: {body.get('model', MODEL)}")
print(f"Tokens: prompt={body['usage']['prompt_tokens']}  completion={body['usage']['completion_tokens']}  total={body['usage']['total_tokens']}")
print("=" * 70)
print(body["choices"][0]["message"]["content"])
print("=" * 70)
