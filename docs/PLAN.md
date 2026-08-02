# PLAN — LoRA Fine-tune gemma-2-2b-it on Modal GPU

Replaces the Colab-T4 + local-merge/gguf flow with a single Modal pipeline.
Identical recipe to the v2 notebook (completion-only loss, LoRA r=16/α=32, 8
epochs, fp16) but runs entirely on Modal GPU.

## Inputs we already have

- `gen_dataset.py` writes `pairs_train.json` / `pairs_holdout.json`.
- `dataset.json` — system/closing/exemplars (template source).
- Runtime template: read from Handy `settings_store.json` (exact 4,158-char prompt used at inference).
- `hf_token.txt` — for gated Gemma weights.

## Environment check

- Modal CLI 1.5.3 present (`modal --version`).
- `.venv` has torch/peft/transformers locally (not needed for Modal, runs remote).
- Modal token assumed configured. Verify with `modal token new` / `modal volume ls` if needed.

## New files

- `modal_finetune.py` — one Modal App, image with HF deps, one Volume mounted so
  all steps share artifacts (`/artifacts` = Volume `gemma2-finetune`).
  - `train()` — LoRA fine-tune (gpu="L4"), saves `lora-adapter` to Volume.
  - `eval_gap()` — term-injection holdout probes (5 gap terms).
  - `merge_local()` — merge LoRA into fp16 base, save merged HF model to Volume.
  - `convert_gguf()` — llama.cpp (built in image): safetensors → GGUF f16 → `Q4_K_M`.
- `run_pipeline()` — orchestrator: train → eval → merge (GGUF convert separately).

## GPU choice

- `train` / `merge` on `L4` (16 GB VRAM, ~$0.60/hr). Gemma-2-2B fits in fp16 (2.6GB),
  LoRA needs no quantization, so we skip bitsandbytes.
- `convert_gguf` on CPU (Modal `cpu`, no GPU needed for llama.cpp quantize).
- Gemma license + HF token needed: mount `hf_token.txt` and log in inside the app.

## Command flow to execute the pipeline

```bash
modal run modal_finetune                      # train -> eval_gap -> merge_local
modal run modal_finetune::convert_gguf        # GGUF f16 + Q4_K_M (separate, CPU)
```

Artifacts land in a Modal Volume mounted at `/artifacts`:
`lora-adapter/`, `gemma2-merged/`, `gemma2-f16.gguf`, `gemma2-q4_k_m.gguf`.

Then locally:

```bash
modal volume get gemma2-finetune /artifacts/gemma2-q4_k_m.gguf ./gemma2-q4_k_m.gguf
```

## Old-method artifacts being deleted

- `gemma2-f16.gguf`, `gemma2-q4_k_m.gguf` — old GGUF models.
- `gemma2-merged/` — old merged model dir (5.2 GB). This step now runs on Modal.
- `lora-adapter/`, `lora-adapter.tgz` — old adapter from Colab.
- `gemma2-lora-finetune-v2.ipynb`, `gemma2-lora-finetune-v2_output.ipynb` — old notebook flow.
- `build_notebook_v2.py` — old notebook builder (superseded by `modal_finetune.py`).
- `merge_local.py` / `convert_gguf.py` — old local/Colab merge+GGUF scripts, replaced
  by `merge()` / `convert_gguf()` Modal steps.

Keep: `gen_dataset.py`, `dataset.json`, `pairs_*.json`, `hf_token.txt`, `.venv`, `Modelfile`.