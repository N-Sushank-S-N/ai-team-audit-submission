# AI Usage Summary

*An honest accounting of where AI assistance accelerated our work, where it produced misleading outputs or hallucinations, and how each issue was caught and verified.*

---

### 1. Where AI Helped
- **Boilerplate and Script Scaffolding:** AI quickly generated boilerplate for HuggingFace `AutoTokenizer` wrappers, command-line argument parsing with `argparse`, and CSV/JSON export logic in `fertility_fixed.py`.
- **Dataset Retrieval Automation:** AI provided the download and tarball extraction logic for the Meta AI FLORES parallel corpus (`prepare_corpus.py`).
- **Grapheme Cluster Regex:** AI correctly pointed out that the Unicode property `\X` in the Python `regex` library (extended grapheme cluster) matches composite Brahmic aksharas (consonant + virama + consonant + vowel matra), enabling a clean `tok/grapheme` denominator.
- **Formula Verification:** AI helped cross-check the algebraic structure of the GQA KV cache footprint ($2 \times L \times N_{kv} \times d_{head} \times 2\text{ bytes}$) and confirmed the arithmetic of $114{,}688\text{ bytes/token}$.

---

### 2. Where AI Misled or Hallucinated (And How It Was Caught)
- **Hallucinated Bug on `unicodedata.normalize("NFC", line)`:**
  - *The Misdirection:* When prompted to find bugs in `fertility.py`, the AI initially claimed that `unicodedata.normalize("NFC")` was a bug that "distorted Devanagari and Dravidian ligatures by altering vowel matras."
  - *How It Was Caught:* We applied the **Evidence Rule** and wrote an isolated test comparing token counts on raw vs. NFC normalized strings across all English, Hindi, Kannada, and Tamil text. The measured delta was **exactly 0.00%**. NFC normalization is standard W3C Unicode canonical composition and is idempotent on standard corpora. The AI's claim was a confident hallucination. We explicitly classified NFC normalization as the "suspicious but harmless" item.
- **Conflating Decimal GB and Binary GiB in VRAM Capacity:**
  - *The Misdirection:* AI initially used $24 \times 1024^3$ (binary GiB) for total GPU memory while simultaneously treating model weights as $8.4 \times 10^9$ (decimal bytes) and overhead as $1.6 \times 10^9$, resulting in a slight mismatch with the benchmark log ($27.2$ sequences predicted vs $25.8$ in log).
  - *How It Was Caught:* We aligned units consistently to decimal GB ($24.0\text{ GB} \times 0.92 = 22.08\text{ GB} - 8.4\text{ GB} - 1.6\text{ GB} = 12.08\text{ GB}$), which yielded $12.08\text{ GB} / 469.76\text{ MB} = 25.72\text{ sequences}$, matching the log's Row 12 ($24 / 0.93 = 25.81\text{ sequences}$) to within $0.3\%$.
- **Overly Optimistic Multilingual SFT Feasibility:**
  - *The Misdirection:* In Part C, the AI initially suggested that Path (a) [SFT] was viable by "fine-tuning with LoRA on synthetic data generated overnight on the A100."
  - *How It Was Caught:* Critical analysis of the constraints revealed the fatal blind spot: **the human reviewer only speaks Hindi and Kannada**, meaning Tamil, Telugu, Bengali, and Marathi would have zero native review. Furthermore, with **$0 external API budget**, all synthetic casualization would have to come from self-distillation of the current 4B model (which is already criticized for being formal and textbook). Distilling poor generations without human review for 4 languages is an unacceptable production risk.
