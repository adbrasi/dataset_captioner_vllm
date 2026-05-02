"""Serve huihui-ai/Huihui-gemma-4-31B-it-abliterated via vLLM on Modal (1x H200).

    modal run modal_gemma4.py::download_model    # uma vez, ~5min
    modal deploy modal_gemma4.py                 # publica em URL pública
"""

import json

import modal

MODEL_NAME = "huihui-ai/Huihui-gemma-4-31B-it-abliterated"
SERVED_NAME = "gemma4-abliterated"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")
    .uv_pip_install("huggingface_hub[hf_xet,hf_transfer]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("gemma4-abliterated")

MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=60 * MINUTES,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME, max_workers=8)
    hf_cache.commit()


@app.function(
    image=vllm_image,
    gpu="H200",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    max_containers=8,
    # min_containers=2,  # descomente pra deixar 2 GPUs sempre warm
)
@modal.concurrent(max_inputs=32, target_inputs=24)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", MODEL_NAME, SERVED_NAME, "llm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.95",
        "--max-num-seqs", "64",
        "--kv-cache-dtype", "fp8",
        "--limit-mm-per-prompt",
        f"'{json.dumps({'image': 4, 'video': 1, 'audio': 0})}'",
        "--mm-processor-kwargs", '\'{"max_soft_tokens": 1120}\'',
        "--async-scheduling",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
