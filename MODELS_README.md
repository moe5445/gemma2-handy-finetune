---
base_model: google/gemma-2-2b-it
license: gemma
language:
  - en
tags:
  - stt
  - post-processing
  - loRA
  - quantization
  - gguf
pipeline_tag: text-generation
---

# gemma2-handy-lora

## What is this?

A **fine-tuned `google/gemma-2-2b-it`** (2B params) that acts as a
**post-processing layer for voice-driven coding tools**. It sits between your
speech-to-text (STT) app and your code.

### The problem it solves

Speech-to-text engines output exactly what you said — full of filler, hedging,
and loose phrasing ("kind of don't hit the server so much maybe?"). Code
commands need to be **one short, unambiguous instruction** that an app can
act on. This model closes that gap.

### What it does

Takes the raw transcription and rewrites it into a **single canonical
engineering instruction in your voice**, picking the precise domain term for
what you meant:

| you say (STT output) | what you meant | model output |
| --- | --- | --- |
| "don't hit the server so much" | rate limiting | `Rate-limit the API calls.` |
| "make sure this only runs once" | idempotency | `Make the webhook handler idempotent.` |
| "load images only when they appear" | lazy loading | `Lazy-load images as they enter the viewport.` |

It classifies input into **11 canonical domains**:
`rate-limit`, `cron`, `mock`, `debounce`, `circuit-breaker`, `optimistic`,
`lazy-load`, `backpressure`, `idempotent`, `blue-green`, `event-sourcing` —
and falls back to fixing spelling/filler when input fits no domain.

### Why a 2B fine-tune instead of a big API model?

- **Private** — runs 100% locally (Ollama / llama.cpp); your voice never
  leaves your machine.
- **Fast + free** — no per-call cost, no network round-trip, ~30 ms on a Mac.
- **Predictable** — constrained to canonical domains, so output is
  app-actionable, not a chatty LLM reply.

## Files

- `gemma2-q4_k_m.gguf` — Q4_K_M quantized model (1.7 GB), ready for
  llama.cpp / Ollama.
- `Modelfile` — Ollama model definition (`ollama create gemma2:handy-lora -f Modelfile`).
- `runtime_template.txt` — prompt template with exemplars; must be used as the
  system prompt at inference (ablation: 23/25 with exemplars vs 17/25 without).

## Training

- Base: `google/gemma-2-2b-it` (gated, accepts Gemma license)
- Method: QLoRA-style LoRA (r=16, α=32, dropout 0.1) on all linear projections
- Data: 794 synthetic pairs (11 terms) / 108 holdout
- Recipe: 4 epochs (early stop patience 1), lr 1e-4 cosine, fp16, batch 1 ×
  grad-accum 8, max_grad_norm 0.5; best checkpoint selected by eval loss
- GPU: single Modal L4 (~30 min)

## Evaluation

25 never-seen real user prompts (verified disjoint from train + holdout):
**23/25** canonical-term hits. Known misses: an "optimistic" like-button update
and an "event-sourcing" order-history phrasing (answers are close but not
canonical).

## Usage

### Ollama

```bash
ollama create gemma2:handy-lora -f Modelfile
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "gemma2:handy-lora",
  "messages": [
    {"role": "system", "content": "<paste runtime_template.txt>"},
    {"role": "user", "content": "don't hit the server so much"}
  ],
  "temperature": 0.2,
  "max_tokens": 20
}'
```

### llama.cpp

```bash
llama-cli -m gemma2-q4_k_m.gguf -p "<prompt>" -n 20 --temp 0.2
```

## Repo

Code, dataset generator, training pipeline, and eval harness:
https://github.com/moe5445/gemma2-handy-finetune
