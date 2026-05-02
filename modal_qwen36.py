"""Serve huihui-ai/Huihui-Qwen3.6-27B-abliterated via vLLM on Modal (1× H200).

Workload alvo: pair captioning de imagens (2 imagens + booru tags + system prompt → JSON).
Qwen 3.6 tem reasoning ativado por padrão — o cliente pode desligar via
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

    modal run modal_qwen36.py::download_model
    modal deploy modal_qwen36.py
"""

import json

import modal

MODEL_NAME = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
SERVED_NAME = "qwen36-abliterated"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")
    .uv_pip_install("huggingface_hub[hf_xet,hf_transfer]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "OMP_NUM_THREADS": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache-qwen36", create_if_missing=True)

app = modal.App("qwen36-abliterated")

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
    scaledown_window=2 * MINUTES,  # 2min = não desperdiça idle entre testes
    timeout=120 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    max_containers=10,    # respeita teto de 10 GPUs simultâneas da conta
)
@modal.concurrent(max_inputs=32, target_inputs=20)  # target=20 -> autoscale mais agressivo
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", MODEL_NAME, SERVED_NAME, "llm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", "65536",
        "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "64",
        "--kv-cache-dtype", "fp8",
        "--limit-mm-per-prompt", json.dumps({"image": 2, "video": 0, "audio": 0}),
        "--mm-processor-kwargs",
        json.dumps({"images_kwargs": {"size": {"longest_edge": 1280, "shortest_edge": 280}}}),
        "--reasoning-parser", "qwen3",
        "--async-scheduling",
    ]
    subprocess.Popen(cmd)
