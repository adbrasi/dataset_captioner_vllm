# runpod_llm — Captioning de pares de imagens com LLMs abliterated

Pipeline de produção pra anotar datasets de pares `_A`/`_B` com modelos abliterated
self-hosted no Modal (vLLM + GPU Hopper). 2 backends prontos: **Gemma 4 31B v2** e
**Qwen 3.6 27B**.

## Arquivos

```
modal_gemma4.py       # serve huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2
modal_qwen36.py       # serve huihui-ai/Huihui-Qwen3.6-27B-abliterated
batch_caption.py      # script de produção (multi-backend, idempotente, resumível)
video_caption.py      # auxiliar — caption de vídeo único (Gemma só)
client_query.py       # cliente single-shot pra 1 imagem
pair_caption.py       # cliente single-shot pra 1 par (debug)
bench_parallel.py     # benchmark de throughput sem thinking
bench_thinking.py     # benchmark com thinking ativo
system_prompt.txt     # system prompt do dataset (NÃO MEXER)
```

## Setup (uma vez)

```bash
cd /home/adolfocesar/projects/runpod_llm

# 1. virtualenv + pacotes
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install modal openai pillow

# 2. autenticar Modal (uma vez por máquina)
.venv/bin/modal token set --token-id <SEU_ID> --token-secret <SEU_SECRET>

# 3. baixar modelos pro Modal Volume (~5min cada, custa centavos)
.venv/bin/modal run modal_gemma4.py::download_model
.venv/bin/modal run modal_qwen36.py::download_model

# 4. publicar os endpoints
.venv/bin/modal deploy modal_gemma4.py
.venv/bin/modal deploy modal_qwen36.py
```

URLs públicas geradas:
- `https://adbrasi--gemma4-abliterated-serve.modal.run`
- `https://adbrasi--qwen36-abliterated-serve.modal.run`

## Estrutura da pasta de input

A pasta deve ter pares de imagem `_A` + `_B` cada um com seu `.txt` de booru tags:

```
dataset/
├── abc123_image_0001_A.jpg     # frame inicial (T0)
├── abc123_image_0001_A.txt     # booru tags da imagem _A
├── abc123_image_0001_B.jpg     # frame final (T1)
├── abc123_image_0001_B.txt     # booru tags da imagem _B
├── abc123_image_0002_A.jpg
├── ...
```

**Convenção de pareamento**: o script descobre por sufixo. Aceita `.jpg/.jpeg/.png/.webp`.
`_A` = imagem 1 do prompt, `_B` = imagem 2. **A ordem importa** — system_prompt assume isso.

## Como rodar (caso de uso principal)

```bash
.venv/bin/python3 batch_caption.py /caminho/da/pasta --backend qwen
```

**Pronto.** Comportamento default:
- Backend: Qwen 3.6 27B abliterated com thinking ON (mais qualidade, ~$15 pra 12.500 pares)
- Concorrência cliente: 64 threads (= 2 GPUs × 32 inflight)
- Modal autoscale até 8 GPUs se a fila crescer
- Idempotente: rodar de novo pula tudo que já foi feito

### Flags úteis

```bash
# limita a 50 pares pra teste rápido
batch_caption.py /pasta --backend qwen --max-pairs 50

# concorrência custom (sweet spot por GPU = 32, então 128 = 4 GPUs)
batch_caption.py /pasta --backend qwen --concurrency 128

# desliga thinking do Qwen (mais rápido, ~$6.50 pra 12.500 pares)
batch_caption.py /pasta --backend qwen --no-thinking

# usa Gemma em vez de Qwen
batch_caption.py /pasta --backend gemma

# log de progresso em local custom
batch_caption.py /pasta --backend qwen --processed-log /tmp/run1.jsonl
```

## Roteiro recomendado pra dataset grande (≥1000 pares)

```bash
# 1. checagem prévia: contar pares e .txt faltantes
.venv/bin/python3 -c "
from pathlib import Path
folder = Path('/caminho/do/dataset')
a = sorted(folder.glob('*_A.jpg'))
b = sorted(folder.glob('*_B.jpg'))
print(f'pares: {len(a)}A + {len(b)}B')
for img in a + b:
    if not img.with_suffix('.txt').exists():
        print(f'  falta .txt: {img.name}')
"

# 2. validação em 50 pares (~3min, ~$0.30) — confirma qualidade do output
.venv/bin/python3 batch_caption.py /caminho/do/dataset \
  --backend qwen --concurrency 128 --max-pairs 50

# inspeciona alguns _B.txt manualmente
head -3 /caminho/do/dataset/*_B.txt | head -20

# 3. produção full — pula automaticamente os 50 já feitos
.venv/bin/python3 batch_caption.py /caminho/do/dataset \
  --backend qwen --concurrency 128
```

**Não precisa chunkar manualmente** — a idempotência (`.processed-qwen.jsonl`) já cobre
crash, Ctrl+C, internet caindo, créditos acabando. Roda o mesmo comando de novo e retoma.

## Saída

Pra cada par processado:
1. `<base>_B.txt` é **sobrescrito** com o `short_prompt` extraído do JSON da resposta
2. Linha registrada em `<pasta>/.processed-<backend>.jsonl` com `{"key", "ts"}`

3 arquivos de controle ficam na pasta:
- **`.booru-cache.json`** — snapshot imutável das booru tags originais. Criado na primeira
  rodada antes de qualquer overwrite. Garantia contra contaminação se tu rodar 2 backends
  no mesmo dataset, ou se _B.txt for sobrescrito por engano.
- **`.processed-gemma.jsonl`** / **`.processed-qwen.jsonl`** — log de pares concluídos
  com `fsync` (durável a kill -9). Logs separados por backend.

### Resume após falha

Mata o processo (Ctrl+C, falha do Modal, internet caiu, sem créditos), depois roda o **mesmo
comando** de novo. O script lê o `.processed-<backend>.jsonl` e pula tudo que já foi feito.

```bash
# crash no meio
^C

# retoma
.venv/bin/python3 batch_caption.py /caminho/da/pasta --backend qwen
# log mostra: "já processados: 1247  |  pendentes: 11253"
```

## Diferenças entre Gemma e Qwen

| | **Gemma 4 31B v2** | **Qwen 3.6 27B** |
|---|---|---|
| Tamanho | 33B params, ~66 GB BF16 | 28B params, ~54 GB BF16 |
| Vision budget | 1.120 soft tokens/imagem (max) | ~16k tokens/imagem em 1080p+ |
| Reasoning parser vLLM | `gemma4` (parsing às vezes vaza) | `qwen3` (rock solid) |
| JSON + thinking | OK mas precisa max_tokens=6000 | OK |
| Throughput com thinking | ~1300 tok/s @ conc=64 | **~1751 tok/s @ conc=64** |
| Latência por par (warm) | ~1.5s | **~0.94s** |
| Cabe sozinho em H100 80GB? | Não (66GB tight) | Sim (54GB tight) |
| **$/12.500 pares (com thinking)** | ~$20 | **~$15** |

**Recomendação**: Qwen como default. Gemma como segundo modelo pra cross-validation
ou quando quiser comparar.

## Custos e scaling

Modal H200 = $4.54/h. Custo por par é **fixo**, independente de quantas GPUs:

| Concorrência | Tok/s agregado | Pares/seg | $/100 pares | $/12.500 pares |
|---|---|---|---|---|
| 16 (1 GPU sub) | 252 | 0.15 | $0.85 | $106 |
| 32 (1 GPU sat) | 749 | 0.41 | $0.31 | $39 |
| 48 (1 GPU max) | 1.149 | 0.64 | $0.20 | $25 |
| **64 (autoscale 2 GPU)** | **1.751** | **1.06** | **$0.12** | **$15** |

Com `--concurrency 128` e `max_containers=8` o Modal pode subir até 8 GPUs em paralelo.
Wall time vira ~6 min, mas **mesmo custo total**: ~$15.

### Quando vale subir mais GPU
- Tu quer terminar rápido: sim, escala até `max_containers=8`
- Tu não tem pressa: deixa em 1 GPU (mais lento, mesmo preço, menos cold starts)

## Configurações do server (modal_*.py)

Ajustes que vão direto no `@app.function` se quiser tunar:

```python
@app.function(
    gpu="H200",                  # mais barato em $/tok pra BF16 27-31B
    scaledown_window=15 * 60,    # 15min sem request → desliga (zera custo)
    timeout=120 * 60,            # tarefa máx 2h por container
    max_containers=8,            # teto de paralelismo
    buffer_containers=1,         # 1 warm durante tráfego (cold start menor)
    # min_containers=2,          # descomenta pra deixar 2 sempre ligadas
)
@modal.concurrent(max_inputs=32, target_inputs=24)
```

- `min_containers=2` paga 2 GPUs idle 24/7 ($217/dia). **Não use** salvo produção
  contínua. Default `min_containers=0` zera quando ocioso.
- `target_inputs=24` faz o autoscaler subir GPU 2 quando a 1 chega em 24/32. Ajuste
  pra mais conservador (ex: `target_inputs=28`) se quiser scale-up mais lento.

## Troubleshooting

### "modal-http: Webhook failed: workspace billing cycle spend limit reached"
Crédito Modal estourou. Vai em `modal.com/settings/<workspace>/billing` e aumenta o
spend limit ou adiciona cartão.

### Cold start muito longo (>5 min)
Normal na primeira chamada após `modal deploy`. vLLM precisa compilar CUDA graphs.
Subsequente é ~30-60s. Se for cancelar/retomar muito, deixa `scaledown_window=300` (5min)
em vez de 15min — rebuilda menos.

### "short_prompt não encontrado" no log de retry
O parsing fallback dá conta na maioria dos casos. Se ficar repetindo num par específico,
provavelmente a imagem ou o booru tag tá problemático. O script falha 3x e segue —
o par fica fora do `.processed-*.jsonl` e tu pode reprocessar manualmente.

### Tempo errado, custo subiu mais que esperado
Cada `.venv/bin/modal deploy` que muda a config dispara cold start na próxima chamada.
Não fica deployando entre testes — o config atual já é o ótimo.

### Quero matar tudo agora
```bash
.venv/bin/modal app stop gemma4-abliterated
.venv/bin/modal app stop qwen36-abliterated
```

## Observabilidade

```bash
# listar apps ativos
.venv/bin/modal app list

# listar containers ativos (cada GPU rodando)
.venv/bin/modal container list

# logs em tempo real
.venv/bin/modal app logs ap-XXXXX  # pega o ID do "modal app list"

# matar container específico (forçar cold start no próximo request)
.venv/bin/modal container stop -y ta-XXXXX
```

Dashboard web: https://modal.com/apps/adbrasi/main/deployed

## Referências

- huihui-ai/Huihui-Qwen3.6-27B-abliterated: https://huggingface.co/huihui-ai/Huihui-Qwen3.6-27B-abliterated
- huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2: https://huggingface.co/huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2
- vLLM Qwen 3.5/3.6 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- vLLM Gemma 4 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- Modal vLLM example: https://modal.com/docs/examples/vllm_inference
