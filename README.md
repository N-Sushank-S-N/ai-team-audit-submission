The layout broke into a single squished line because non-breaking spaces and collapsed linebreaks made GitHub's Markdown parser choke on the directory tree and the quickstart commands.

Here is a rebuilt, highly readable version with your live Desmos link integrated:

```markdown
# 🔍 AI Team Audit: Tokenizer Economics & Serving Capacity Reconciliation

[![Submission Status](https://img.shields.io/badge/Audit_Status-Complete_%26_Verified-success.svg)](#)
[![Defense Ready](https://img.shields.io/badge/Defense_Session-Ready-blue.svg)](#)
[![Desmos Model](https://img.shields.io/badge/Desmos-Interactive_Model-orange.svg)](https://www.desmos.com/calculator/vkpiolfqte)

A forensic audit of `REPORT_v0.md` and the LLM serving stack for the **AI Team Intern Assignment — The Audit**.

---

## 📌 Executive Summary of Findings

| Section | Previous Intern Claim (`REPORT_v0.md`) | Audit Finding & Proven Reality |
| :--- | :--- | :--- |
| **Part A: Tokenizer Economics** | Hindi is **5.89× to 7.0× worse** than English due to an "inherent property of the script." Budget 6× serving cost. | **False.** The previous intern tested an English-only tokenizer (`gpt2`). On a 1,012-sentence parallel evaluation corpus (FLORES-101) using an Indic-aware tokenizer (`xlm-roberta-base`), Hindi requires only **1.25×** the tokens of English; Dravidian languages (Kannada, Tamil) require **1.35×**. The true serving cost premium is ~25%–35%, not 500%. |
| **Part A: Metric & Denominator** | Used whitespace words (`tok/word`) and Unicode codepoints (`tok/char`). | Agglutinative Dravidian languages fuse inflections, inflating `tok/word` (2.58 in Kannada vs 1.40 in English) while using fewer words per sentence. **Tokens per Parallel Sentence** is the only reliable metric holding semantic information constant. |
| **Part B: KV Cache Concurrency** | Not computed analytically. | **114,688 bytes (112 KiB)** per token exactly. Maximum concurrent 4096-token sequences on a 24GB L4 GPU is **25.72** (matches benchmark log Row 12: $24 / 0.93 = \mathbf{25.81}$ sequences). |
| **Part B: Long-Prompt Scaling** | Assumed throughput scales linearly to batch 48 (~3200 tok/s). | **Throughput collapses after batch 24** (1607 tok/s $\to$ 1384 $\to$ 1298) because the workload exceeds the ~25.8 sequence limit, triggering **7 and 23 sequence preemptions**. |
| **Part B: Goodput Misreading** | Reported 1607 tok/s serving throughput at batch 24. | Misread `reported_tok_s` (total prefill + decode tokens/s). Honest generation goodput is only **~201 gen tok/s** (Way 1: $12288 / 61.16\text{s}$) to **~250 gen tok/s** (Way 2: $24 / 0.09607\text{s}$ ITL)—an ~8× exaggeration. |
| **Part C: Tone Strategy** | Three open paths (SFT, Rewriter, Prompting). | Recommend **Path (c) Prompt Engineering** with strict kill criteria. SFT and Rewriter fail because our single reviewer speaks only Hindi and Kannada (leaving 4 languages unverified) and a $0 API budget prohibits clean synthetic data generation. |

---

## 📊 Interactive Visual Model (Desmos)

An interactive graph modeling the KV Cache Memory Ceiling and Preemption Cliff is live:

👉 **[Launch Interactive Desmos Calculator](https://www.desmos.com/calculator/vkpiolfqte)**

* **Key Parameters Visualized:**
  * Memory ceiling: $L = 28, N_{kv} = 8, d = 128, p = 2$ (FP16) $\to 114{,}688\text{ bytes/token}$ ($112\text{ KiB}$).
  * Available KV pool on 24GB L4 = **$12.08\text{ GB}$**.
  * Concurrency ceiling at sequence length 4096 = **$25.72\text{ sequences}$**.
  * Visualizes the vertical preemption wall at $x = 25.72$ where throughput drops from 1607.4 tok/s and contrasts reported throughput against honest generation goodput (200.9 tok/s).

---

## 🗂️ Deliverables & Navigation

* 📓 **[`NOTEBOOK.md`](NOTEBOOK.md):** Graded chronological lab notebook documenting all 7 phases of hypotheses, experiments, measured results, surprises, and dead ends.
* 🤖 **[`AI_USAGE.md`](AI_USAGE.md):** Transparent disclosure of where AI tools assisted and where AI hallucinations were caught and corrected (e.g. proving NFC normalization was harmless).
* 📄 **[`partA/memo.md`](partA/memo.md):** Executive recommendation memo on multilingual tokenizer economics and routing.
* 📐 **[`partB/calculations.md`](partB/calculations.md):** Step-by-step mathematical proofs and log reconciliations for B1, B2, B3, and B4.
* 🎯 **[`partC/memo.md`](partC/memo.md):** Decision memo recommending the prompt engineering path across 6 Indic languages under tight resource constraints.

---

## 🏗️ Repository Architecture

```text
ai-team-audit-submission/
├── README.md               # Overview, executive summary, and reproduction guide
├── NOTEBOOK.md             # Graded chronological lab notebook
├── AI_USAGE.md             # Transparent AI usage and verification disclosure
├── partA/
│   ├── data/               # 1,012 parallel sentences each (FLORES-101: eng, hin, kan, tam)
│   ├── prepare_corpus.py   # Corpus download and alignment verification script
│   ├── fertility_audit.py  # Empirical proof of each flaw under the Evidence Rule
│   ├── fertility_fixed.py  # Multi-tokenizer, multi-denominator benchmark script
│   ├── results/            # Raw JSON and CSV benchmark outputs (gpt2 vs xlm-roberta-base)
│   └── memo.md             # Part A4 Executive Recommendation Memo
├── partB/
│   ├── calculations.md     # Mathematical proofs for B1, B2, B3, B4
│   └── verify_bench.py     # Standalone Python verifier for serving log numbers
└── partC/
    └── memo.md             # Part C Strategic Decision Memo

```

---

## 🚀 Reproduction & Quickstart

### Prerequisites

```bash
pip install transformers tiktoken regex sentencepiece pyarrow

```

### 1. Re-derive Part A2 Flaws (Evidence Rule)

Demonstrates the isolated impact of `line.split(" ")` empty strings (+1.4% to +2.0%), macro vs. micro averaging (-1.0%), casing distortions (+3.1%), and tokenizer vocabulary swaps (-80.8%):

```bash
python partA/fertility_audit.py ../starter_kit/corpus_sample

```

### 2. Run Cross-Language Benchmark (1,012 Parallel Sentences)

Evaluates 5 denominators across English, Hindi, Kannada, and Tamil:

```bash
# Baseline GPT-2
python partA/fertility_fixed.py \
  --corpus eng=partA/data/eng_eval.txt \
  --corpus hin=partA/data/hin_eval.txt \
  --corpus kan=partA/data/kan_eval.txt \
  --corpus tam=partA/data/tam_eval.txt \
  --tokenizer gpt2

# Multilingual Indic-Aware Tokenizer (XLM-RoBERTa)
python partA/fertility_fixed.py \
  --corpus eng=partA/data/eng_eval.txt \
  --corpus hin=partA/data/hin_eval.txt \
  --corpus kan=partA/data/kan_eval.txt \
  --corpus tam=partA/data/tam_eval.txt \
  --tokenizer hf:xlm-roberta-base

```

### 3. Re-derive Part B Serving Arithmetic

Independently verifies KV cache footprint ($114{,}688\text{ bytes/tok}$), concurrency ceiling ($25.72\text{ seqs}$), preemption mechanism, and honest goodput (~$201\text{ gen tok/s}$):

```bash
python partB/verify_bench.py ../starter_kit/bench/bench_log.csv

```

```

```
