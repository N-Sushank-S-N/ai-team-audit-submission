# Executive Recommendation Memo: Indic Tokenization & Serving Cost Audit

**To:** AI Leadership & Infrastructure Capacity Planning  
**From:** Antigravity Audit Team  
**Date:** September 2026  
**Subject:** Corrected Multilingual Tokenizer Economics and Traffic Routing Strategy  

---

### 1. Corrected Headline Numbers
The previous report (`REPORT_v0.md`) claimed that Hindi tokenization is **5.89× to 7.0× worse** than English, asserting that serving Indic traffic would cost ~6× more due to an inherent "property of the script." 

Both the diagnosis and cost projections were incorrect. The intern evaluated an English-only BPE tokenizer (`gpt2`), whose vocabulary lacks Indic characters, causing severe byte fallback shredding. When recomputed properly on a 1,012-sentence parallel evaluation corpus (FLORES devtest) across Indo-Aryan (Hindi) and Dravidian (Kannada, Tamil) languages with modern tokenizers, the true token overhead is modest:

| Language | GPT-2 (tok/sent) | GPT-2 Expansion | XLM-R (tok/sent) | XLM-R Expansion | XLM-R (tok/word) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **English (eng)** | 26.72 | 1.00× | 30.30 | 1.00× | 1.40 |
| **Hindi (hin)** | 198.31 | **7.42×** | 37.77 | **1.25×** | 1.49 |
| **Kannada (kan)** | 363.01 | **13.58×** | 40.97 | **1.35×** | 2.58 |
| **Tamil (tam)** | 415.19 | **15.54×** | 40.86 | **1.35×** | 2.47 |

**Core Finding:** Across all Indic and Dravidian languages, the true semantic token expansion with an Indic-aware tokenizer is only **1.25× to 1.35×** relative to English, **not 6×**. 

---

### 2. Routing & Capacity Recommendation
1. **Do NOT spin up a dedicated, isolated Indic serving cluster based on assumed 6× costs.** Budget an infrastructure compute overhead of only **~25% to 35%** for Indic requests compared to English.
2. **Enforce Multilingual / Indic-Aware Tokenizers in Front-End Routing:** If the main foundation model uses an English-dominated vocabulary (like GPT-2 or older LLaMA-1/2), route Indic requests to a multilingual backbone with a balanced vocabulary (e.g., Llama-3.2 with 128k vocab, Gemma-2 with 256k vocab, or XLM-R/Sarvam). Using an English-centric tokenizer on Indic traffic wastes 7× to 15× GPU compute on pure byte overhead.
3. **Capacity Dimensioning:** In Dravidian languages, tokens-per-word appears inflated (2.58 tok/word in Kannada vs 1.40 in English) due to linguistic agglutination (inflections fused into single orthographic words). However, because Dravidian sentences require fewer total words to convey identical semantic information, the end-to-end token footprint is only 1.35× English. Dimension capacity strictly using **Tokens per Request (Semantic Equivalence)**, never whitespace words.

---

### 3. The Biggest Caveat
**Domain and Stylistic Discrepancy (FLORES vs Production Conversational Traffic):**  
FLORES-101 is a professionally translated, formal, standardized text corpus. Production user traffic in India predominantly features:
- **Code-mixing / Hinglish / Tanglish / Kanglish:** Users frequently mix Latin script with native phonetics (e.g., *"Aaj meeting kab hai?"*).
- **Colloquial slang and informal orthography:** Omission of formal matras, phonetic abbreviations, and regional dialects.
Latin-script Indic code-mixing tokenizes differently than pure native Devanagari/Dravidian scripts and can alter token efficiency depending on whether the tokenizer was trained on Romanized Indic data.

---

### 4. Production Metric to Monitor
**Metric:** `P95 Generated Tokens per User Request (by Language Code)` & `Prompt-to-Completion Expansion Ratio`.  
**Why:** If production queries begin exhibiting token counts >1.5× relative to English baseline requests for comparable intent categories, it immediately flags either:
1. Routing regressions (e.g., requests erroneously falling back to an English-centric tokenizer), or
2. High code-mixing causing vocabulary fragmentation in production prompts.
