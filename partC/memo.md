# Executive Decision Memo: Indic Conversational Tone Strategy

**To:** Product Leadership & AI Steering Committee  
**From:** Antigravity Audit Team  
**Date:** September 2026  
**Subject:** Strategy Recommendation for Casualizing Indic Assistant Responses  

---

### Executive Recommendation
We recommend **Path (c): Prompt-Engineering Only**, executed via a phased, language-tiered rollout. 

Under our strict constraints (no external API budget, 1 reviewer for only 2 of the 6 languages, and a 3-week deadline), Path (a) [SFT] and Path (b) [Rewriter] are unviable and present unacceptable production risks. Path (c) requires zero architectural changes, introduces zero serving latency, avoids distilling unvetted synthetic hallucinations, and can be empirically validated on Day 1.

---

### Assumptions
1. **Human Evaluation Bottleneck:** The single native speaker can review **only Hindi and Kannada** for 10 hours/week. Tamil, Telugu, Bengali, and Marathi (4 of the 6 languages) have **zero native human review capability** before launch.
2. **Compute & API Constraints:** We have 1× NVIDIA A100-80GB for 2 weeks (~336 GPU hours) and **$0 external API budget** (prohibiting the use of frontier APIs like GPT-4o for synthetic data generation or automated LLM judging).
3. **Serving Latency & Cost Invariance:** The product team will not accept added round-trip latency or VRAM overhead in production serving clusters.
4. **Primary Failure Mode:** In low-resource Indic generation, synthetic casualization easily degenerates into grammatical corruption, English code-switching errors, or loss of factual accuracy.

---

### Back-of-Envelope Arithmetic

#### 1. Reviewer Throughput & Capacity
- Evaluating a conversational response pair (reading prompt, comparing formal vs. casual response, checking grammatical naturalness and factual fidelity) requires **~3 minutes per item** = **20 evaluations per hour**.
- Total reviewer capacity over 3 weeks: $10\text{ hours/week} \times 3\text{ weeks} = \mathbf{30\text{ hours}} = \mathbf{600\text{ total evaluations}}$.
- Divided equally between available languages: **300 evaluations for Hindi**, **300 evaluations for Kannada**, and **0 evaluations for Tamil, Telugu, Bengali, Marathi**.

#### 2. Training & Data Generation vs Serving Arithmetic
- **Path (a) SFT:** Requires at least 2,500 high-quality casual pairs per language across 6 languages = 15,000 pairs. On our single A100, generating 15,000 synthetic responses with the local 4B model takes ~30 hours. However, without external APIs or native reviewers for 4 languages, >66% of training data would be unverified self-distilled text. Fine-tuning a 4B model on A100 takes ~40 GPU hours, leaving almost zero time for iteration or hyperparameter tuning.
- **Path (b) Inference-Time Rewriter (≤1B):** Adding a second sequential model introduces an extra autoregressive decode pass per request (~30–50% latency increase, ~150–250ms additional TTFT/ITL overhead). It also requires loading two models onto serving GPUs, reducing available KV cache and cutting concurrent serving capacity by ~25%.
- **Path (c) Prompt Engineering:** Requires **0 training hours**, **0 added serving latency**, and **0 extra VRAM**. The 336 A100 GPU hours are repurposed entirely for high-throughput batch generation of prompt variations and automated syntactic validation.

---

### Success Metric & Numeric Threshold
- **Metric:** Blinded Pairwise Win-Rate on Conversational Naturalness (Hindi & Kannada) against the production formal baseline across a curated eval set of 100 conversational multi-turn prompts.
- **Numeric Threshold:** **$\ge 65\%$ human reviewer preference** for the casual prompt over the formal baseline, with a mandatory **$\le 2\%$ factual error / hallucination rate**.
- **Secondary Metric:** Perplexity and length ratio invariance (casual response token length within $0.85\times$ to $1.15\times$ of baseline, ensuring no degenerative repetition loops).

---

### Kill Criterion
- **Phase 1 Kill (Day 7 / End of Week 1):** If after testing the top 4 prompt engineering templates, the human reviewer win rate on Hindi or Kannada is **$< 55\%$** (indistinguishable from noise) OR the factual hallucination rate exceeds **$5\%$**, immediately abandon the casual prompt update.
- **Phase 2 Kill (Day 14 / End of Week 2):** For Tamil, Telugu, Bengali, and Marathi: if automated safety filters, length heuristics, or toxic language scans detect anomalous token distributions (>1.4× token bloat or repetition), **kill the release for those 4 unreviewed languages** and launch casual mode strictly as a targeted Beta for Hindi and Kannada.

---

### First Experiment (Day 1)
1. **Morning (Hours 0–4):** Formulate 4 targeted prompt variants in the system prompt:
   - *Variant A:* Zero-shot persona prompt with explicit tone guidelines (avoiding textbook honorifics like "कृपया" / "ಆಶಿಸುತ್ತೇನೆ", encouraging conversational tone).
   - *Variant B:* 3-shot in-context learning with colloquial discourse markers (e.g., natural sentence connectors, everyday pronouns).
   - *Variant C:* Explicit negative constraints ("Do not sound like a government notice or textbook").
   - *Variant D:* Hybrid (Negative constraints + 2 conversational few-shot exemplars).
2. **Afternoon (Hours 4–8):** Run batch inference on the local A100 across 25 Hindi and 25 Kannada representative conversational queries for all 4 variants + baseline (250 total responses, takes <15 minutes on A100).
3. **Deliverable for Day 2:** Deliver a blinded, randomized 50-pair evaluation sheet to the native reviewer for their initial 2.5-hour review block to establish the winning prompt candidate by end of Day 2.
