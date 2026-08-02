# gemma2-handy-finetune

A tiny fine-tune that sits **between your speech-to-text (STT) app and your code**.
It takes the messy transcription of what you said and rewrites it into **one short,
domain-exact engineering instruction**.

| you speak (STT output) | model output |
| --- | --- |
| "don't hit the server so much" | `Rate-limit the API calls.` |
| "make sure this only runs once" | `Make the webhook handler idempotent.` |
| "load images only when they appear" | `Lazy-load images as they enter the viewport.` |

The model is a **LoRA fine-tune of `google/gemma-2-2b-it`**, quantized to
**Q4_K_M (1.7 GB)** and served locally with **Ollama**. No API calls, no cloud.

It classifies input into 11 canonical domains:
`rate-limit`, `cron`, `mock`, `debounce`, `circuit-breaker`, `optimistic`,
`lazy-load`, `backpressure`, `idempotent`, `blue-green`, `event-sourcing`.

---

## Folder layout

```
├── data/
│   ├── pairs_train.json / pairs_holdout.json   # generated dataset (794 / 108)
│   ├── runtime_template.txt                    # prompt template WITH exemplars (also the Handy prompt)
│   ├── runtime_clean.txt / runtime_clean_tpl.txt  # exemplar-free variants (for ablations)
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

### 0. Get a HuggingFace token + accept the Gemma license (one time)

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

### 1. Generate the dataset (local)

```bash
.venv/bin/python finetune/gen_dataset.py
```

Hand-written seed pairs per term, expanded with synonym swaps into 794 train /
108 holdout pairs. Deterministic (SEED 42).

### 2. Train on Modal

```bash
modal run finetune/modal_finetune.py::train
```

LoRA r=16 / α=32 / dropout 0.1, 4 epochs (early stop patience 1), lr 1e-4, fp16
on a Modal L4. All artifacts land in the `gemma2-finetune` volume. The
checkpoint with the **lowest eval loss** (not the last epoch) is promoted to
`lora-adapter`.

### 3. Evaluate on never-seen probes

```bash
modal run finetune/modal_finetune.py::eval_fresh
```

Every probe is verified to be absent from train + holdout first. Current score:
**23/25** with the full template vs **17/25** without exemplars — the exemplars
in the prompt matter, keep them.

### 4. Merge + quantize

```bash
modal run finetune/modal_finetune.py::merge_local
modal run finetune/modal_finetune.py::convert_gguf
```

Merges the adapter into the fp16 base, converts to GGUF f16, then Q4_K_M.
Download the result:

```bash
modal volume get gemma2-finetune /artifacts/gemma2-q4_k_m.gguf models/gemma2-q4_k_m.gguf --force
```

### 5. Serve with Ollama

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

---

## Current model status

- Trained: 794 pairs, 11 terms, LoRA on gemma-2-2b-it, ~30 min on one L4.
- Eval: 23/25 on `fresh_probes.txt` (25 never-seen real prompts).
- Shipped: `models/gemma2-q4_k_m.gguf` (1.7 GB), served as `gemma2:handy-lora` in Ollama.
- Known misses: "optimistic" like-button update and "event-sourcing" order history
  (model answers are close but not canonical — candidates for a future retrain).
