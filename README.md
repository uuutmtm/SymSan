# SymSan

Code for **"SymSan: Mitigating Malware Embedding in Large Language Models via
Parameter Space Symmetry"** (CCS 2026).


## Install

```bash
conda env create -f environment.yml && conda activate SymSan
```

## Quick start

```bash
# Sanitize and verify
python pipeline.py --model /path/to/Llama-2-7b-chat-hf
```



## Usage examples

```bash
python pipeline.py --model /path/to/model \
    --save /path/to/save
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--model` | required | model path  |
| `--seed` | `42` |  |
| `--device` | `cuda` | torch device, e.g. `cuda:1` |
| `--mmlu-samples` | `0` (off) | MMLU samples, measured before and after |
| `--save` | off | directory for the sanitized model + tokenizer |
| `--no-verify` | off | skip equivalence verification and coverage audit |
| `--output-dir` | `results` | where to write `symsan_summary.json` |
#
