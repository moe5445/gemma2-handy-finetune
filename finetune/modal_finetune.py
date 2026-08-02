import os
import subprocess
import sys

import modal

MODEL_ID = "google/gemma-2-2b-it"
MAXLEN = 2048

VOL = "/artifacts"
ARTIFACTS = modal.Volume.from_name("gemma2-finetune", create_if_missing=True)

app = modal.App("gemma2-q2b-finetune")

# ---------------------------------------------------------------------------
# Training image: HF stack + the small local inputs mounted in.
# ---------------------------------------------------------------------------
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers[torch]",
        "peft",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "huggingface_hub",
    )
    .add_local_file("data/runtime_template.txt", "/inputs/runtime_template.txt")
    .add_local_file("data/runtime_clean.txt", "/inputs/runtime_clean.txt")
    .add_local_file("data/runtime_clean_tpl.txt", "/inputs/runtime_clean_tpl.txt")
    .add_local_file("data/pairs_train.json", "/inputs/pairs_train.json")
    .add_local_file("data/pairs_holdout.json", "/inputs/pairs_holdout.json")
    .add_local_file("hf_token.txt", "/inputs/hf_token.txt")
    .add_local_file("data/fresh_probes.txt", "/inputs/fresh_probes.txt")
)

# Conversion image: llama.cpp built once into the image; tokenizer.model fetched
# from HF at run time (needs the token too, so re-use train data by mounting).
convert_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "g++", "build-essential")
    .pip_install("torch", "numpy~=1.26.4", "transformers==4.57.6", "sentencepiece>=0.1.98,<0.3.0", "gguf>=0.1.0", "protobuf>=4.21.0,<5.0.0", "huggingface_hub")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp",
        "cd /root/llama.cpp && cmake -B build"
        " -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF",
        "cd /root/llama.cpp && cmake --build build --target llama-quantize -j$(nproc)",
    )
    .add_local_file("hf_token.txt", "/inputs/hf_token.txt")
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _login():
    from huggingface_hub import login

    token = open("/inputs/hf_token.txt").read().strip()
    login(token=token)
    return token


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def _load_pairs(template_file="runtime_template.txt"):
    import json

    tpl = open(f"/inputs/{template_file}").read()
    train = json.load(open("/inputs/pairs_train.json"))
    holdout = json.load(open("/inputs/pairs_holdout.json"))
    return tpl, train, holdout


def _make_prompt(user_input, tpl):
    return tpl.replace("${output}", user_input)


def _enc_pair(tokenizer, prompt, answer):
    p = tokenizer(prompt, truncation=True, max_length=MAXLEN)["input_ids"]
    a = tokenizer(answer + " ", truncation=True, max_length=MAXLEN)["input_ids"]
    if len(p) + len(a) > MAXLEN:
        p = p[: MAXLEN - len(a)]
    ids = p + a
    return {"input_ids": ids, "labels": [-100] * len(p) + a}


SYNO = {
    "blue-green": ["blue", "green"],
    "rate-limit": ["rate", "throttl", "per second", "per minute", "cap"],
    "cron": ["cron", "schedul", "timer", "3am"],
    "mock": ["mock", "stub", "fake", "canned"],
    "lazy-load": ["lazy", "defer", "on scroll", "viewport"],
    "idempotent": ["idempot", "duplicate", "twice", "dedup"],
    "backpressure": ["backpressure", "back pressure", "slow down"],
    "debounce": ["debounce"],
    "circuit-breaker": ["circuit break", "trip", "cooldown"],
    "event-sourcing": ["event sourcing", "event log", "append-only"],
    "optimistic": ["optimistic", "roll back", "revert"],
}


def _hit(reply, term, canonical):
    low = reply.lower()
    if term in low or canonical.lower() in low:
        return True
    return any(s in low for s in SYNO.get(term, []))


# ---------------------------------------------------------------------------
# Best-checkpoint selection: scan lora-out/checkpoint-*, pick the one with the
# lowest eval_loss (the "recommended" checkpoint), and which we promote to
# lora-adapter. Logs the full eval-loss history so the choice is auditable.
# ---------------------------------------------------------------------------
def _best_checkpoint_path():
    import glob
    import json
    import os

    ckpts = sorted(
        glob.glob(f"{VOL}/lora-out/checkpoint-*"),
        key=lambda p: int(p.rsplit("-", 1)[1]),
    )
    rows = []
    for c in ckpts:
        state = json.load(open(os.path.join(c, "trainer_state.json")))
        evals = [h["eval_loss"] for h in state["log_history"] if "eval_loss" in h]
        rows.append({"checkpoint": os.path.basename(c), "eval_loss": evals[-1] if evals else None, "epoch": state.get("epoch")})
    for r in rows:
        print(f"  candidate {r['checkpoint']}: eval_loss={r['eval_loss']!s:>8} epoch={r['epoch']}")
    if not rows:
        raise RuntimeError("no checkpoints found under lora-out/")
    best = min(rows, key=lambda r: r["eval_loss"])
    print("RECOMMENDED checkpoint:", best["checkpoint"], "(lowest eval_loss)")
    return os.path.join(VOL, "lora-out", best["checkpoint"])


@app.function(image=train_image, gpu="L4", volumes={VOL: ARTIFACTS}, timeout=1800)
def promote_best():
    import os
    import shutil

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    _login()
    src = _best_checkpoint_path()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base, src)
    model.save_pretrained(f"{VOL}/lora-adapter")
    tok = _tokenizer()
    tok.save_pretrained(f"{VOL}/lora-adapter")
    ARTIFACTS.commit()
    print("promoted", src, "->", f"{VOL}/lora-adapter")


# ---------------------------------------------------------------------------
# 1. QLoRA fine-tune (Modal GPU)
# ---------------------------------------------------------------------------
@app.function(image=train_image, gpu="L4", volumes={VOL: ARTIFACTS}, timeout=3600)
def train():
    import json

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset as TD
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

    _login()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tpl, train_pairs, hold_pairs = _load_pairs()
    train_recs = [_enc_pair(tokenizer, _make_prompt(r["user"], tpl), r["output"]) for r in train_pairs]
    hold_recs = [_enc_pair(tokenizer, _make_prompt(r["user"], tpl), r["output"]) for r in hold_pairs]
    print("train:", json.dumps({"n_train": len(train_recs), "n_holdout": len(hold_recs)}))

    class TokDS(TD):
        def __init__(self, recs):
            self.recs = recs

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            r = self.recs[i]
            return {"input_ids": torch.tensor(r["input_ids"]), "labels": torch.tensor(r["labels"])}

    def collate(batch):
        max_in = max(b["input_ids"].numel() for b in batch)
        ids = torch.zeros(len(batch), max_in, dtype=torch.long)
        labs = torch.full((len(batch), max_in), -100, dtype=torch.long)
        for bi, b in enumerate(batch):
            n = b["input_ids"].numel()
            ids[bi, :n] = b["input_ids"]
            labs[bi, :n] = b["labels"]
        return {"input_ids": ids, "labels": labs}

    n_steps = (len(train_recs) * 4) // (1 * 8)
    args = TrainingArguments(
        output_dir=f"{VOL}/lora-out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        num_train_epochs=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.06 * n_steps)),
        weight_decay=0.01,
        fp16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        seed=0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        max_grad_norm=0.5,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=TokDS(train_recs),
        eval_dataset=TokDS(hold_recs),
        data_collator=collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()

    # Promote the checkpoint with the lowest holdout eval_loss — NOT the final
    # epoch. Early stopping may keep extra checkpoints; pick the best one.
    best = _best_checkpoint_path()
    base_best = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    from peft import PeftModel

    best_model = PeftModel.from_pretrained(base_best, best)
    best_model.save_pretrained(f"{VOL}/lora-adapter")
    tokenizer.save_pretrained(f"{VOL}/lora-adapter")
    ARTIFACTS.commit()
    print("adapter saved from recommended checkpoint:", best)


# ---------------------------------------------------------------------------
# 2. Eval on fresh probes: real user phrasings NEVER seen in train or holdout.
#    Single probe set: data/fresh_probes.txt (25 probes, all 11 terms).
#    Always uses the full exemplar template (data/runtime_template.txt).
# ---------------------------------------------------------------------------
@app.function(image=train_image, gpu="L4", volumes={VOL: ARTIFACTS}, timeout=600)
def eval_fresh():
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _login()
    tpl, _, _ = _load_pairs()
    probes = json.load(open("/inputs/fresh_probes.txt"))

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    tok_ev = AutoTokenizer.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, f"{VOL}/lora-adapter")
    model.eval()

    def gen(prompt, max_new=64):
        enc = tok_ev(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        return tok_ev.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    results = []
    for i, p in enumerate(probes):
        reply = gen(_make_prompt(p["user"], tpl))
        hit = _hit(reply, p["term"], p["canonical_line"])
        results.append({"i": i, "term": p["term"], "hit": hit, "reply": reply[:120], "canonical": p["canonical_line"]})
        print(f"[{i:02d}:{p['term']:14s}] hit={hit}  ->  {reply[:120]}")

    p = sum(1 for v in results if v["hit"])
    print("PASS", p, "/", len(results), "on fresh_probes.txt")
    print(json.dumps(results, indent=1))
    return results


# ---------------------------------------------------------------------------
# 3. Merge adapter into base fp16
# ---------------------------------------------------------------------------
@app.function(image=train_image, gpu="L4", volumes={VOL: ARTIFACTS}, timeout=1800)
def merge_local():
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _login()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base.eval()
    print("base loaded, fp16")

    model = PeftModel.from_pretrained(base, f"{VOL}/lora-adapter", is_trainable=False)
    model.to(torch.float16)
    model = model.merge_and_unload()
    model.eval()
    print("merged OK")

    out = f"{VOL}/gemma2-merged"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(out)
    for fn in sorted(os.listdir(out)):
        print(f"  {fn}: {round(os.path.getsize(os.path.join(out, fn))/1e6, 1)} MB")
    ARTIFACTS.commit()
    print("saved to", out)


# ---------------------------------------------------------------------------
# 4. GGUF conversion: safetensors -> f16 -> Q4_K_M (CPU, llama.cpp in image)
# ---------------------------------------------------------------------------
@app.function(image=convert_image, volumes={VOL: ARTIFACTS}, timeout=3600)
def convert_gguf():
    import os

    from huggingface_hub import hf_hub_download

    _login()
    merged = f"{VOL}/gemma2-merged"
    if not os.path.exists(os.path.join(merged, "tokenizer.model")):
        hf_hub_download("google/gemma-2-2b-it", "tokenizer.model", local_dir=merged)
        print("fetched tokenizer.model")

    f16 = f"{VOL}/gemma2-f16.gguf"
    out = f"{VOL}/gemma2-q4_k_m.gguf"

    def run(cmd, **kw):
        print(">>>", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
        if r.stdout:
            print(r.stdout[-1200:])
        if r.returncode != 0:
            print("STDERR", r.stderr[-1200:])
            raise SystemExit(r.returncode)
        return r

    if not os.path.exists(f16):
        run([sys.executable, "/root/llama.cpp/convert_hf_to_gguf.py", merged, "--outfile", f16, "--outtype", "f16"])
        print("f16 gguf built:", round(os.path.getsize(f16) / 1e6, 1), "MB")
    else:
        print("f16 gguf exists")

    if not os.path.exists(out):
        run(["/root/llama.cpp/build/bin/llama-quantize", f16, out, "Q4_K_M"])
    else:
        print("q4 gguf exists")
    print("Q4 gguf:", round(os.path.getsize(out) / 1e9, 2), "GB")
    ARTIFACTS.commit()
    print("DONE")


# ---------------------------------------------------------------------------
# Orchestration: train -> fresh eval -> merge (GGUF conversion is a separate
# call). The eval runs on real user phrasings never seen in train/holdout.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def run_pipeline():
    train.remote()
    eval_fresh.remote()
    merge_local.remote()