# SymSan

Code for **"SymSan: Mitigating Malware Embedding in Large Language Models via
Parameter Space Symmetry"** (CCS 2026).

## Abstract
Despite the considerable advantages of open-source large languagemodels (LLMs) in terms of interpretability and customizability, the full-access sharing paradigm inevitably exposes them to a severesecurity threat, i.e., malware embedding attacks (MEAs), where adversaries embed malware binaries (i.e., payloads) into model parameters for subsequent execution. While existing sanitization defenses have been explored to eliminate payloads within parameters, they exhibit limited generalization against emerging MEAs and lack theoretical guarantees for model utility preservation. In this paper, we propose SymSan, an effective LLM sanitization method grounded in parameter space symmetry theory. SymSan leverages rigorously designed permutation and scaling symmetries, systematically transforming parameter matrices across complementary component pairs, thereby eliminating embedded payloads while ensuring mathematical functional equivalence. Experiments across diverse LLMs against a wide spectrum of MEAs demonstrate that SymSan achieves at least 0.47 BER against all evaluated existing MEAs while maintaining equivalent model performance on both foundational and downstream tasks, significantly outperforming all baselines.

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
