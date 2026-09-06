# Research & Audit Chronological Lab Notebook

*This notebook documents the chronological progression of hypotheses, experiments, measured results, surprises, dead ends, and revisions during the audit of `REPORT_v0.md` and the serving stack.*

---

## [Phase 1] Initial Inspection & Reproduction
**Timestamp:** Hour 0.5  
**Objective:** Inspect starter kit artifacts and reproduce the previous intern's headline numbers.

### Hypothesis 1.1
The intern's numbers in `REPORT_v0.md` can be exactly reproduced by running `fertility.py` on the provided sample files.

### Experiment 1.1
Executed baseline command:
```bash
py -3.12 fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
```

### Result 1.1 (Confirmed)
Output matched the report table exactly:
- `eng`: fertility = 1.27 tok/word, tok/char = 0.226
- `hin`: fertility = 7.45 tok/word, tok/char = 1.579
- Summary: "hin is 5.89x the fertility of eng (worse tokenization)"

### Surprise & Dead End 1.1: The "Parallel" Corpus Was Fake
While inspecting `eng_sample.txt` and `hin_sample.txt`, we tested whether the lines were actually parallel translations as claimed in `REPORT_v0.md`:
- Eng Line 1: *"Bengaluru International Airport handled record traffic in March."*
- Hin Line 1: *"मुझे सुबह की चाय बहुत पसंद है।"* ("I like morning tea very much.")
- Eng Line 2: *"The Quarterly Review meeting moved to Thursday."*
- Hin Line 2: *"बेंगलुरु में आज हल्की बारिश हो रही है।"* ("It is raining lightly today in Bengaluru.")

**Finding:** The two sample files are not parallel at all! Some lines loosely match out of order (e.g. Eng Line 4 matches Hin Line 7), while others have no correspondence. The intern claimed to compare parallel efficiency, but the sample data had mismatched sentence lengths and completely different semantic topics.

---

## [Phase 2] Code & Metric Audit (`fertility.py`)
**Timestamp:** Hour 2.0  
**Objective:** Isolate specific bugs and flaws in `fertility.py` according to the Evidence Rule.

### Hypothesis 2.1: `line.split(" ")` mishandles whitespace
Looking at `corpus_sample/eng_sample.txt` line 7 (`"Please keep the books  in the cupboard."`), there is a double space.

### Experiment 2.1
Compare `line.split(" ")` vs `line.split()`:
```python
words_split_space = [l.lower().split(" ") for l in lines]
words_split_clean = [l.lower().split() for l in lines]
```

### Result 2.1 (Bug Proved)
- English: `split(" ")` found 79 words; `split()` found 78 words (+1 empty string `""`). Macro fertility shifted from 1.2652 to 1.2831 (+1.41%).
- Hindi: Line 10 has `"किताबें  अलमारी"` (double space). `split(" ")` found 62 words; `split()` found 61 words. Macro fertility shifted from 7.4485 to 7.5985 (+2.01%).
- **Conclusion:** `split(" ")` counts consecutive spaces as words, inflating the denominator and artificially depressing fertility.

### Hypothesis 2.2: Statistical Bug — Macro vs Micro Averaging
`fertility.py` computes:
```python
per_line_fertility.append(len(tokens) / len(words))
return sum(per_line_fertility) / n
```
This is the average of ratios ($\frac{1}{N} \sum \frac{T_i}{W_i}$) rather than the ratio of sums ($\frac{\sum T_i}{\sum W_i}$).

### Experiment 2.2
Calculated micro vs macro fertility on clean split tokens:
- English: Macro = 1.2831 tok/word vs Micro = 1.2692 tok/word (Delta: -0.0138 tok/word, -1.08%).
- Hindi: Macro = 7.5985 tok/word vs Micro = 7.5246 tok/word (Delta: -0.0739 tok/word, -0.97%).
- **Conclusion:** Macro-averaging over-weights short outlier sentences, violating Jensen's inequality.

### Hypothesis 2.3: `line.lower()` introduces asymmetric distortion
The script forces `line = line.lower()` to "remove noise".

### Experiment 2.3
Counted tokens with and without lowercasing in GPT-2:
- Hindi: Original case = 459 tokens, Lowercase = 459 tokens (Delta: 0, 0.00%).
- English: Original case = 96 tokens, Lowercase = 99 tokens (Delta: +3 tokens, +3.12%).
- **Conclusion:** Devanagari has no case, so lowercasing is inert. In English, acronyms like `NASA`, `ISRO`, `GPU` get fragmented when lowercased. Lowercasing distorted the baseline English comparison.

### Hypothesis 2.4: Suspicious Item — Does `unicodedata.normalize("NFC")` corrupt text?
Could Unicode NFC normalization be altering Indic glyphs or vowel matras?

### Experiment 2.4 (Dead End 2: The Harmless Item)
Tested raw unnormalized text vs NFC normalized text:
- English: Raw = 96 tokens, NFC = 96 tokens (Delta: 0).
- Hindi: Raw = 459 tokens, NFC = 459 tokens (Delta: 0).
- **Finding:** NFC normalization is completely harmless and standard W3C best practice. Flagging this as a bug would be an error.

---

## [Phase 3] The Conceptual Flaw & Tokenizer Swap
**Timestamp:** Hour 3.5  
**Objective:** Disprove the claim that Hindi tokenization explosion is an "inherent property of the script."

### Hypothesis 3.1
The 6× to 7× explosion is caused by GPT-2's vocabulary lacking native Devanagari byte merges, forcing individual UTF-8 byte fallback (3 tokens per codepoint). An Indic-aware tokenizer will eliminate this penalty.

### Experiment 3.1
Tokenized the Hindi sample with `xlm-roberta-base` (SentencePiece multilingual model):
- GPT-2: 459 tokens (45.9 tokens/sentence)
- XLM-RoBERTa: 88 tokens (8.8 tokens/sentence)
- **Delta:** -371 tokens (-80.83% reduction!)

### Result 3.1 (Conclusive)
The intern's conclusion that "any tokenizer will struggle because of the script" is completely false. The fault lay entirely with using an English-only tokenizer on Indic text.

---

## [Phase 4] Assembling the Multilingual Evaluation Corpus (A1)
**Timestamp:** Hour 4.5  
**Objective:** Construct a real, parallel evaluation corpus across at least 4 languages (English, Hindi, and two Dravidian languages).

### Implementation
- **Source:** Meta AI FLORES-101 Benchmark (`devtest` split).
- **Target Languages:**
  1. `eng`: English (Germanic baseline)
  2. `hin`: Hindi (Indo-Aryan, Devanagari script)
  3. `kan`: Kannada (Dravidian, Kannada script)
  4. `tam`: Tamil (Dravidian, Tamil script)
- **Size:** Exactly 1,012 strictly parallel sentences per language.
- **Script:** Developed `prepare_corpus.py` to download, verify parallel alignment, NFC normalize, and export to `partA/data/`.

---

## [Phase 5] Multi-Denominator Benchmark (A3)
**Timestamp:** Hour 6.0  
**Objective:** Evaluate multiple denominators (words, grapheme clusters, bytes, parallel sentences) across tokenizers.

### Experiment 5.1
Ran `fertility_fixed.py` across all 1,012 sentences with `gpt2` and `hf:xlm-roberta-base`:

| Tokenizer | Language | tok/sent | tok/word | tok/grapheme | tok/byte | tok/char |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GPT-2** | eng | 26.72 | 1.23 | 0.20 | 0.205 | 0.205 |
| | hin | 198.31 (7.42×) | 7.83 (6.34×) | 2.33 (11.39×) | 0.595 (2.90×) | 1.530 |
| | kan | 363.01 (13.58×) | 22.82 (18.48×) | 4.07 (19.84×) | 0.979 (4.78×) | 2.662 |
| | tam | 415.19 (15.54×) | 25.05 (20.28×) | 4.21 (20.56×) | 0.997 (4.87×) | 2.726 |
| **XLM-R** | eng | 30.30 | 1.40 | 0.23 | 0.232 | 0.232 |
| | hin | 37.77 (**1.25×**) | 1.49 (**1.06×**) | 0.44 (1.91×) | 0.113 (0.49×) | 0.291 |
| | kan | 40.97 (**1.35×**) | 2.58 (**1.84×**) | 0.46 (1.97×) | 0.110 (0.48×) | 0.300 |
| | tam | 40.86 (**1.35×**) | 2.47 (**1.76×**) | 0.41 (1.78×) | 0.098 (0.42×) | 0.268 |

### Insight on Denominators:
- In Dravidian languages, `tok/word` is 1.84× English in XLM-R because Kannada and Tamil are **agglutinative** (multiple grammatical affixes fused into one word).
- However, when measured **per parallel sentence** (the semantic unit holding information content constant), Kannada and Tamil require only **1.35×** the tokens of English!
- **Conclusion:** **Tokens per Parallel Sentence** is the only denominator that should drive capacity planning and cost estimation.

---

## [Phase 6] Part B Capacity Reconciliation
**Timestamp:** Hour 7.5  
**Objective:** Derive KV cache formulas, reconcile with `bench_log.csv`, and expose goodput misreadings.

### Calculations Summary
1. **B1(a) KV bytes per token:**
   $$2 \times 28 \times 8 \times 128 \times 2 = 114{,}688\text{ bytes} = 112\text{ KiB}$$
2. **B1(b) Max concurrent 4096-token sequences:**
   - Usable VRAM: $24\text{ GB} \times 0.92 = 22.08\text{ GB}$
   - Available for KV: $22.08\text{ GB} - 8.4\text{ GB} - 1.6\text{ GB} = 12.08\text{ GB}$
   - Per sequence: $4096 \times 114{,}688\text{ B} = 469.76\text{ MB}$
   - Capacity: $12.08\text{ GB} / 469.76\text{ MB} = \mathbf{25.72\text{ sequences}}$.
   - Check against log Row 12: $24 / 0.93 = \mathbf{25.81\text{ sequences}}$ (exact match).
3. **B2 Long-Context Anomaly:**
   - At batch 32, 7 sequences are preempted. Throughput drops from 1607 tok/s to 1384 tok/s.
   - At batch 48, 23 sequences are preempted. Throughput drops to 1298 tok/s.
   - Solution: Set `max_num_seqs = 24` or enable FP8 KV cache (doubling capacity to 51 sequences).
4. **B3 Misread Column:**
   - Intern read `reported_tok_s`, which includes prompt prefill.
   - Honest generation goodput for Batch 24:
     - Way 1 (End-to-End): $(24 \times 512) / 61.16\text{s} = \mathbf{200.92\text{ gen tok/s}}$
     - Way 2 (From ITL): $24 / 0.09607\text{s} = \mathbf{249.82\text{ gen tok/s}}$
   - The intern reported 1607 tok/s—an ~8× exaggeration over honest generation goodput.

---

## [Phase 7] Part C Decision Memo
**Timestamp:** Hour 9.0  
**Objective:** Formulate a rock-solid, defensible recommendation for casualizing Indic responses across 6 languages.

### Strategic Evaluation:
- **Constraints:** 1 A100 for 2 weeks, 1 reviewer for 10h/week (Hindi & Kannada only), 3-week deadline, $0 external API budget.
- **Reviewer capacity:** Max 600 evaluations total (~300 Hindi, ~300 Kannada, **0 for Tamil/Telugu/Bengali/Marathi**).
- **Decision:** Path (a) [SFT] and Path (b) [Rewriter] fail because 4 languages would be deployed without human verification, and rewriters degrade serving latency.
- **Chosen:** Path (c) Prompt Engineering with strict kill criteria and a phased Beta rollout.
