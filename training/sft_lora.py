#!/usr/bin/env python
"""P5: LoRA SFT with trl + peft. Real run target = a small smoke (Qwen2.5-0.5B-Instruct);
switch base_model to the local Qwen2.5-7B for a full QLoRA run on the A100.

    python training/sft_lora.py --config config/sft_lora.yaml

Reports answer-accuracy before vs after training (numeric-fact match on held-in QA),
saves the LoRA adapter, and writes outputs/sft_metrics.json. A LLaMA-Factory equivalent
config (config/llamafactory_sft.yaml) is provided as the production-scale path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from prepare_data import SYSTEM, _context  # sibling import (script dir on sys.path)

from agentic_rag.config import get_settings
from agentic_rag.eval.generation_metrics import answer_accuracy
from agentic_rag.ingestion import load_documents
from agentic_rag.synthesis import load_qa


def build_eval_examples(settings, n: int):
    docs = {d.doc_id: d for d in load_documents(settings)}
    qas = load_qa(settings)[:n]
    out = []
    for qa in qas:
        user = f"资料：\n{_context(qa, docs)}\n\n问题：{qa.question}"
        out.append(([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}], qa.answer))
    return out


def eval_model(model, tok, examples, max_new_tokens: int = 160) -> float:
    import torch

    model.eval()
    accs = []
    with torch.no_grad():
        for msgs, gold in examples:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            accs.append(answer_accuracy(gold, gen))
    return round(sum(accs) / len(accs), 4) if accs else 0.0


def load_dataset_text(tok, sharegpt_path: str):
    from datasets import Dataset

    from agentic_rag.io_utils import load_jsonl

    rows = load_jsonl(sharegpt_path)
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    return Dataset.from_dict({"text": texts})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sft_lora.yaml")
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.base_model:
        cfg["base_model"] = args.base_model
    if args.max_steps:
        cfg["max_steps"] = args.max_steps

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    settings = get_settings()
    base = cfg["base_model"]
    print(f"[sft] base model: {base}")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Single-GPU sets device_map; under `accelerate launch` (FSDP, WORLD_SIZE>1, P8) accelerate
    # places/shards the model and we skip generation-based eval (needs a param gather).
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if not distributed:
        model_kwargs["device_map"] = {"": 0}
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)

    eval_examples = [] if distributed else build_eval_examples(settings, cfg.get("eval_samples", 8))
    acc_before = None if distributed else eval_model(model, tok, eval_examples)
    if not distributed:
        print(f"[sft] answer-accuracy BEFORE: {acc_before}")

    lora = LoraConfig(
        r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05), bias="none", task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg.get("per_device_batch", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        max_steps=cfg.get("max_steps", 60),
        logging_steps=cfg.get("logging_steps", 5),
        bf16=cfg.get("bf16", True),
        max_seq_length=cfg.get("max_seq_length", 1024),
        dataset_text_field="text",
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        report_to=[],
        save_strategy="no",
    )
    ds = load_dataset_text(tok, cfg["sharegpt_path"])
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    acc_after = None if distributed else eval_model(trainer.model, tok, eval_examples)
    delta = round(acc_after - acc_before, 4) if (acc_before is not None and acc_after is not None) else None
    metrics = {
        "base_model": base, "max_steps": cfg.get("max_steps", 60),
        "answer_accuracy_before": acc_before, "answer_accuracy_after": acc_after,
        "delta": delta, "eval_samples": len(eval_examples),
        "distributed": distributed, "adapter_dir": str(out_dir),
    }
    metrics_path = settings.output_dir / "sft_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not distributed:
        print(f"[sft] answer-accuracy AFTER:  {acc_after}  (Δ {delta:+})")
    print(f"[sft] adapter -> {out_dir} | metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
