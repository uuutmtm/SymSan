# SymSan

Code for **"SymSan: Mitigating Malware Embedding in Large Language Models via
Parameter Space Symmetry"** (CCS 2026).


## Install

```bash
conda env create -f environment.yml && conda activate SymSan
```

## Quick start

```bash
# Sanitize and verify (paper default: r = 0.5)
python pipeline.py --model /path/to/Llama-2-7b-chat-hf
```



## Usage examples

```bash
python pipeline.py --model /path/to/model \
    --save /path/to/save

# Also measure MMLU accuracy before/after (needs MMLU, see Data)
python pipeline.py --model /path/to/Llama-2-13b-chat-hf --mmlu-samples 5

# Larger scaling range r (factors drawn from [1-r, 1+r])
python pipeline.py --model /path/to/model --scale-range 0.8

```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--model` | required | model path  |
| `--scale-range` | `0.5` | r; scaling factors in [1-r, 1+r] |
| `--seed` | `42` |  |
| `--device` | `cuda` | torch device, e.g. `cuda:1` |
| `--mmlu-samples` | `0` (off) | MMLU samples, measured before and after |
| `--save` | off | directory for the sanitized model + tokenizer |
| `--no-verify` | off | skip equivalence verification and coverage audit |
| `--output-dir` | `results` | where to write `symsan_summary.json` |
#
