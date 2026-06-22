#!/usr/bin/env python
"""P6: GRPO post-training with trl. Rewards reuse the P4 eval signals (training-evaluation
closed loop). Smoke default = Qwen2.5-0.5B-Instruct, a handful of steps.

    python training/grpo_train.py --config config/grpo.yaml

A veRL equivalent config (config/verl_grpo.yaml) is provided as the production-scale path
named by the roadmap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from prepare_data import SYSTEM, _context
from rewards import ALL_REWARDS
from sft_lora import build_eval_examples, eval_model

from agentic_rag.config import get_settings
from agentic_rag.ingestion import load_documents
from agentic_rag.synthesis import load_qa


def build_grpo_dataset(settings, tok, n: int):
    from datasets import Dataset

    docs = {d.doc_id: d for d in load_documents(settings)}
    rows = []
    for qa in load_qa(settings)[:n]:
        ctx = _context(qa, docs)
        user = f"资料：\n{ctx}\n\n问题：{qa.question}"
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False)
        rows.append({"prompt": prompt, "answer": qa.answer, "evidence": ctx})
    return Dataset.from_list(rows)


def _reward_trend(log_history) -> dict:
    rewards = [e["reward"] for e in log_history if "reward" in e]
    if not rewards:
        return {}
    return {"reward_first": round(rewards[0], 4), "reward_last": round(rewards[-1], 4),
            "reward_delta": round(rewards[-1] - rewards[0], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/grpo.yaml")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps:
        cfg["max_steps"] = args.max_steps

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    settings = get_settings()
    base = cfg["base_model"]
    print(f"[grpo] base model: {base}")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)

    eval_examples = build_eval_examples(settings, cfg.get("eval_samples", 8))
    acc_before = eval_model(model, tok, eval_examples)
    print(f"[grpo] answer-accuracy BEFORE: {acc_before}")

    lora = LoraConfig(
        r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    grpo_cfg = GRPOConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg.get("per_device_batch", 8),
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        max_steps=cfg.get("max_steps", 20),
        logging_steps=cfg.get("logging_steps", 2),
        bf16=cfg.get("bf16", True),
        num_generations=cfg.get("num_generations", 4),
        max_prompt_length=cfg.get("max_prompt_length", 640),
        max_completion_length=cfg.get("max_completion_length", 160),
        temperature=cfg.get("temperature", 0.9),
        beta=cfg.get("beta", 0.04),
        report_to=[],
        save_strategy="no",
    )
    ds = build_grpo_dataset(settings, tok, cfg.get("n_samples", 24))
    trainer = GRPOTrainer(model=model, reward_funcs=ALL_REWARDS, args=grpo_cfg,
                          train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))

    acc_after = eval_model(trainer.model, tok, eval_examples)
    metrics = {
        "base_model": base, "max_steps": cfg.get("max_steps", 20),
        "num_generations": cfg.get("num_generations", 4),
        "answer_accuracy_before": acc_before, "answer_accuracy_after": acc_after,
        "delta": round(acc_after - acc_before, 4),
        **_reward_trend(trainer.state.log_history),
    }
    metrics_path = settings.output_dir / "grpo_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[grpo] answer-accuracy AFTER: {acc_after} (Δ {metrics['delta']:+}) | {_reward_trend(trainer.state.log_history)}")
    print(f"[grpo] adapter -> {out_dir} | metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
