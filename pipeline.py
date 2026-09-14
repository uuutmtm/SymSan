"""SymSan pipeline: sanitize a single model and verify functional equivalence.

Applies SymSan to one Hugging Face causal LM, verifies that the sanitized
model is functionally equivalent to the original (same next-token
predictions, bounded logit deviation), optionally measures MMLU accuracy
before/after, and optionally saves the sanitized model.

Example:
    python pipeline.py --model model/Mistral-7B-Instruct-v0.3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make repo-local modules importable regardless of the caller's cwd
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from defenses.symmetry import run_symmetry_defense
from validator.mmlu_val import test_mmlu

def mmlu_data_dir() -> str:
    """Locate the MMLU CSVs: data/val (legacy layout) or data/validation
    (current cais/mmlu layout)."""
    base = os.path.join(current_dir, "datasets", "mmlu", "data")
    for sub in ("val", "validation"):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            return d
    return os.path.join(base, "val")


def resolve_model_path(name: str) -> str:
    """Accept an absolute/relative path, or a name under the model root."""
    if os.path.exists(name):
        return name
    root = os.environ.get("SYMSAN_MODEL_ROOT", os.path.join(current_dir, "model"))
    candidate = os.path.join(root, name)
    if os.path.exists(candidate):
        return candidate
    return name  # let transformers report the original path problem


def load_model(path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Some models (e.g. Gemma3) require bfloat16 to avoid NaN in float16
    model_dtype = torch.float16
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", "").lower()
        text_cfg = getattr(cfg, "text_config", {})
        if isinstance(text_cfg, dict):
            cfg_dtype = text_cfg.get("dtype", "")
        else:
            cfg_dtype = getattr(text_cfg, "dtype", "")
        if "bfloat16" in str(cfg_dtype) or "gemma" in model_type:
            model_dtype = torch.bfloat16
            print(f"[INFO] Detected {model_type}: using bfloat16 for numerical stability")
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=model_dtype, device_map=device, trust_remote_code=True,
    )
    return model, tokenizer


def run_mmlu(model, tokenizer, num_samples: int):
    data_dir = mmlu_data_dir()
    if not os.path.isdir(data_dir):
        print(f"[WARN] MMLU data not found at {data_dir}, skipping evaluation")
        return None
    try:
        return test_mmlu(model, tokenizer, data_dir=data_dir, num_samples=num_samples)
    except Exception as e:
        print(f"[WARN] MMLU evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="SymSan pipeline (single model)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path or name of the model to sanitize (resolved under "
                             "$SYMSAN_MODEL_ROOT or ./model when a bare name is given)")
    parser.add_argument("--scale-range", type=float, default=0.5,
                        help="Scaling range r; factors drawn from [1-r, 1+r] "
                             "(paper default 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip functional-equivalence verification")
    parser.add_argument("--mmlu-samples", type=int, default=0,
                        help="MMLU samples per subject before/after sanitization "
                             "(0 = skip; requires MMLU under datasets/)")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save the sanitized model and tokenizer")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Where to write symsan_summary.json")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)

    print("=" * 60)
    print("  SymSan pipeline")
    print(f"  model        : {model_path}")
    print(f"  scale_range  : {args.scale_range}, seed {args.seed}")
    print("=" * 60)

    # 1. Load model
    print("[INFO] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model(model_path, args.device)
    load_s = time.time() - t0
    print(f"[INFO] Model loaded in {load_s:.1f}s")

    # 2. Optional utility evaluation before sanitization
    mmlu_before = run_mmlu(model, tokenizer, args.mmlu_samples) if args.mmlu_samples > 0 else None

    # 3. Sanitize
    model, report = run_symmetry_defense(
        model, tokenizer,
        seed=args.seed,
        scale_range=args.scale_range,
        verify=not args.no_verify,
    )

    # 4. Optional utility evaluation after sanitization
    mmlu_after = run_mmlu(model, tokenizer, args.mmlu_samples) if args.mmlu_samples > 0 else None

    # 5. Optional save of the sanitized model
    if args.save:
        print(f"[INFO] Saving sanitized model to {args.save}")
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)

    # 6. Machine-readable summary
    summary = {
        "model": model_path,
        "scale_range": args.scale_range,
        "seed": args.seed,
        "device": args.device,
        "load_time_s": round(load_s, 2),
        "mmlu_before": mmlu_before,
        "mmlu_after": mmlu_after,
        "defense_report": report,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "symsan_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"  Summary written to {out_path}")
    if report:
        eq = report.get("equivalence")
        if eq:
            print(f"  Equivalence : {eq.get('status')} "
                  f"(max logit diff {eq.get('max_diff', float('nan')):.2e}, "
                  f"tokens match: {eq.get('tokens_match')})")
        print(f"  Defense time: {report.get('total_time_s', 0):.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
