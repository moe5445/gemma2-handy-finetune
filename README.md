# gemma2-handy-finetune

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

### How it was built

Trained with LoRA (r=16/α=32) on 794 hand-written + synonym-augmented
training pairs across the 11 domains; evaluated on 25 real user prompts the
model had never seen (**23/25** correct). Quantized to **Q4_K_M (1.7 GB)** —
no GPU needed at inference.

---

## Folder layout

## Folder layout

```
├── data/
│   ├── pairs_train.json / pairs_holdout.json   # generated dataset (794 / 108)
│   ├── runtime_template.txt                    # prompt template WITH exemplars (also the Handy prompt)
│   ├── fresh_probes.txt                        # 25 real-user probes never seen in training
│   └── dataset.json                            # source system/closing/exemplars
├── models/
│   └── gemma2-q4_k_m.gguf                      # final quantized model (1.7 GB)
├── finetune/
│   ├── gen_dataset.py                          # builds pairs_train/holdout from hand-written seeds
│   └── modal_finetune.py                       # Modal app: train, eval, merge, convert
├── Modelfile                                   # Ollama model definition
├── hf_token.txt                                # your HuggingFace token (gitignored)
└── docs/PLAN.md                                # original design notes
```

---

## How the pipeline works

```
gen_dataset.py  ->  modal train  ->  eval_fresh  ->  merge  ->  GGUF  ->  Ollama  ->  Handy/STT app
  (794 pairs)      (LoRA, L4)     (real probes)  (fp16)     (Q4_K_M)   (local)
```

### 0. Set up Modal CLI + account (one time)

All training, eval, merge and GGUF conversion run on Modal — you need an
account and the CLI on your machine:

1. Create a free account at https://modal.com/ (sign in with GitHub).
2. Install the CLI — Python 3.9+ required:

   ```bash
   pip install modal
   # or: pipx install modal
   ```

3. Authenticate — this opens your browser and links the CLI to your account:

   ```bash
   modal token new
   ```

4. Verify everything works and the artifact volume exists:

   ```bash
   modal volume ls gemma2-finetune   # should list lora-out, lora-adapter, gemma2-q4_k_m.gguf
   ```

   > GPU note: `train`/`merge`/`eval` use an NVIDIA L4 GPU. Modal's free tier
   > does **not** include GPUs — add a payment method under
   > https://modal.com/settings/billing (you are billed per-second while a GPU
   > runs, roughly $0.60/hr for an L4; a full train run costs ~$1).

### 1. Get a HuggingFace token + accept the Gemma license (one time)

Gemma-2 is a gated model — you must accept its license and authenticate before
any step that downloads it from HuggingFace (train, eval, merge, convert):

1. Create an account at https://huggingface.co/join (skip if you have one).
2. Accept the license: open https://huggingface.co/google/gemma-2-2b-it and
   click **Agree and access repository** on the license card.
3. Create a token: https://huggingface.co/settings/tokens → **Create new token**
   → type `Read` → copy the `hf_...` string.
4. Save it in this repo as `hf_token.txt` (one line, no quotes). The file is
   gitignored and mounted into the Modal image at `/inputs/hf_token.txt`:

   ```bash
   echo "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > hf_token.txt
   ```

   Verify locally: `huggingface-cli login --token $(cat hf_token.txt)` should
   print "Login successful". Skip the step if you already did it before —
   the token never expires by default.

### 2. Generate the dataset (local)

```bash
.venv/bin/python finetune/gen_dataset.py
```

Hand-written seed pairs per term, expanded with synonym swaps into 794 train /
108 holdout pairs. Deterministic (SEED 42).

### 3. Train on Modal

```bash
modal run finetune/modal_finetune.py::train
```

LoRA r=16 / α=32 / dropout 0.1, 4 epochs (early stop patience 1), lr 1e-4, fp16
on a Modal L4. All artifacts land in the `gemma2-finetune` volume. The
checkpoint with the **lowest eval loss** (not the last epoch) is promoted to
`lora-adapter`.

### 4. Evaluate on never-seen probes

```bash
modal run finetune/modal_finetune.py::eval_fresh
```

Every probe is verified to be absent from train + holdout first. Current score:
**23/25** with the full template vs **17/25** without exemplars — the exemplars
in the prompt matter, keep them.

### 5. Merge + quantize

```bash
modal run finetune/modal_finetune.py::merge_local
modal run finetune/modal_finetune.py::convert_gguf
```

Merges the adapter into the fp16 base, converts to GGUF f16, then Q4_K_M.
Download the result:

```bash
modal volume get gemma2-finetune /artifacts/gemma2-q4_k_m.gguf models/gemma2-q4_k_m.gguf --force
```

### 6. Serve with Ollama

```bash
ollama create gemma2:handy-lora -f Modelfile
ollama serve   # listens on http://localhost:11434
```

---

## Hooking it up to any STT app (like Handy)

The pattern is always the same — the fine-tuned model is a **post-processing
step after transcription**:

```
microphone -> STT (transcription text) -> [this model rewrites it] -> clean command -> app acts
```

### Install Ollama + Handy (first time)

1. **Ollama** (serves the model locally):
   - Download and install from https://ollama.com/ (or `brew install ollama`).
   - Register the fine-tuned model (needs `models/gemma2-q4_k_m.gguf` on disk):

     ```bash
     ollama create gemma2:handy-lora -f Modelfile
     ollama serve            # runs on http://localhost:11434 (started by default)
     ```

   - Sanity check: `ollama list` shows `gemma2:handy-lora`.

2. **Handy** (the STT app): download and install from https://handy.computer/
   (free; runs transcription fully on your Mac).

### Option A — via Ollama (any app)

Send the transcription with the prompt template as the system message:

```bash
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "gemma2:handy-lora",
  "messages": [
    {"role": "system", "content": "'"$(cat data/runtime_template.txt)"'"},
    {"role": "user", "content": "don't hit the server so much"}
  ],
  "temperature": 0.2,
  "max_tokens": 20
}'
```

Or from Python:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
template = open("data/runtime_template.txt").read()

def post_process(transcription: str) -> str:
    resp = client.chat.completions.create(
        model="gemma2:handy-lora",
        messages=[
            {"role": "system", "content": template},
            {"role": "user", "content": transcription},
        ],
        temperature=0.2,
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip()
```

The template contains a `${output}` placeholder — **replace it with the
transcription** if you build the prompt yourself (Ollama handles it if you put
the transcription in the user message and the template in the system message).

### Option B — Handy-specific

Handy already has a post-processing slot wired to your local Ollama:

1. Quit Handy.
2. Edit `~/Library/Application Support/com.pais.handy/settings_store.json`:
   - `settings.post_process_provider_id` = `custom` (points at `http://localhost:11434/v1`)
   - `settings.post_process_models.custom` = `gemma2:handy-lora`
   - `settings.post_process_prompts[0]` (id `domain_expert_code`): paste the
     content of `data/runtime_template.txt`
3. Start Handy and use **Transcribe with Post-Processing** (default
   `option+shift+space`).

> Handy rewrites `settings_store.json` when it quits — apply edits **after**
> quitting, and re-apply if they get reverted. The repo keeps no live copy;
> `data/runtime_template.txt` is the source of truth.

### Option C — prompt an agent to connect it for you

Paste this into any AI agent (or Handy's settings) to automate the whole
connection:

```text
Connect my local fine-tuned model to Handy and Ollama:

1. Ollama must be installed (https://ollama.com/). If `ollama` is not on
   PATH, install it, then run: ollama create gemma2:handy-lora -f Modelfile
   (Modelfile is at /Users/moe/gemma2-handy-finetune/Modelfile and points at
   models/gemma2-q4_k_m.gguf). Verify with `ollama list`.

2. Handy must be installed from https://handy.computer/. Quit Handy first.

3. Edit ~/Library/Application Support/com.pais.handy/settings_store.json:
   - settings.post_process_enabled = true
   - settings.post_process_provider_id = "custom"
   - settings.post_process_models.custom = "gemma2:handy-lora"
   - In settings.post_process_prompts, find the entry with
     id = "domain_expert_code" and replace its prompt field with the full
     contents of /Users/moe/gemma2-handy-finetune/data/runtime_template.txt
     (4,885 chars, includes ${output} placeholder).
   - settings.post_process_selected_prompt_id = "domain_expert_code"

4. Relaunch Handy. Post-processing now routes transcriptions through the
   local model at http://localhost:11434/v1.
```

---

## Current model status

- Trained: 794 pairs, 11 terms, LoRA on gemma-2-2b-it, ~30 min on one L4.
- Eval: 23/25 on `fresh_probes.txt` (25 never-seen real prompts).
- Shipped: `models/gemma2-q4_k_m.gguf` (1.7 GB), served as `gemma2:handy-lora` in Ollama.
- Download the quantized model from HuggingFace Hub:
  https://huggingface.co/momoe5445/gemma2-handy-lora (GGUF + Modelfile + template + model card).
- Code: https://github.com/moe5445/gemma2-handy-finetune
- Known misses: "optimistic" like-button update and "event-sourcing" order history
  (model answers are close but not canonical — candidates for a future retrain).
