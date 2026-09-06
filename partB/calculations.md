# Part B: Capacity Reconciliation and Serving Stack Audit

This document provides exact, step-by-step mathematical derivations and empirical reconciliation between the hardware/model specification (`bench/model_spec.md`) and the load-test execution trace (`bench/bench_log.csv`).

---

## B1. KV Cache Footprint & Maximum Concurrency

### (a) Exact KV-Cache Bytes per Token

From `bench/model_spec.md`:
- **Model:** FLM-4B-Instruct
- **Layers ($L$):** 28
- **KV Attention Heads ($N_{kv}$):** 8 (Grouped-Query Attention / GQA)
- **Head Dimension ($d_{head}$):** 128
- **Precision:** fp16 (2 bytes per scalar element)

In autoregressive transformer decoding, each token attends to all prior keys and values. For each layer and each token position, the engine must store one Key vector and one Value vector across all $N_{kv}$ heads:

$$\text{Elements per token per layer} = 2 \times N_{kv} \times d_{head} = 2 \times 8 \times 128 = 2{,}048\text{ elements}$$

Summing over all $L = 28$ layers:
$$\text{Total scalar elements per token} = 28 \times 2{,}048 = 57{,}344\text{ elements}$$

At fp16 precision ($2\text{ bytes/element}$):
$$\text{KV bytes per token} = 57{,}344 \times 2 = \mathbf{114{,}688\text{ bytes}} = \mathbf{112\text{ KiB}}\text{ (EXACTLY)}$$

$$\left(\frac{114{,}688}{1{,}048{,}576} \approx 0.109375\text{ MiB per token}\right)$$

---

### (b) Approximate Maximum Concurrent 4096-Token Sequences

To determine how many concurrent sequences of length $S = 4096$ can fit in GPU VRAM:

1. **Total GPU VRAM:** 1× NVIDIA L4 = $24\text{ GB}$.
2. **Serving Engine Usable Memory:**
   $$\text{Usable VRAM} = 24\text{ GB} \times \text{gpu\_memory\_utilization} (0.92) = 22.08\text{ GB}$$
3. **Model Weights Footprint:**
   $$4.2\text{ B parameters} \times 2\text{ bytes (fp16)} = 8.40\text{ GB}$$
4. **Non-KV Runtime Overhead:**
   $$\text{Activations, CUDA graphs, PagedAttention metadata} = 1.60\text{ GB}$$
5. **Remaining Memory Dedicated to KV Cache Pool:**
   $$\text{KV Pool Memory} = 22.08\text{ GB} - 8.40\text{ GB} - 1.60\text{ GB} = \mathbf{12.08\text{ GB}}\text{ }(12{,}080{,}000{,}000\text{ bytes})$$
6. **KV Cache Footprint per 4096-Token Sequence:**
   $$\text{Bytes per Sequence} = 4096\text{ tokens} \times 114{,}688\text{ bytes/token} = \mathbf{469{,}762{,}048\text{ bytes}} \approx 469.76\text{ MB}\text{ }(448\text{ MiB})$$
7. **Theoretical Maximum Concurrent Sequences:**
   $$\text{Max Sequences} = \frac{12{,}080{,}000{,}000\text{ bytes}}{469{,}762{,}048\text{ bytes/seq}} \approx \mathbf{25.72\text{ sequences}}$$

Thus, the GPU can hold approximately **25 to 26 concurrent 4096-token sequences**.

#### Checking Against `bench_log.csv`:
Examine **Row 12** (`batch_size=24, prompt_len=3584, gen_len=512`, where total sequence length = $3584 + 512 = 4096$ tokens):
- The log reports `kv_cache_util = 0.93` (93.0% peak KV cache utilization).
- Implied total KV capacity:
  $$\text{Capacity} = \frac{24\text{ sequences}}{0.93} = \mathbf{25.81\text{ sequences}}$$
- **Reconciliation:** The analytical prediction from the specification alone ($25.72$ sequences) matches the observed benchmark capacity ($25.81$ sequences) to within $0.3\%$.

---

## B2. Long-Context Throughput Anomaly & Mechanism

### 1. Identifying the Anomaly
In the long-context sweep (`prompt_len = 3584, gen_len = 512`), naive scaling expectations dictate that total throughput should increase monotonically with batch size until saturating compute/bandwidth:

| Row | Batch Size | Wall Clock (s) | Reported tok/s | ITL p50 (ms) | E2E p95 (ms) | Preempted Seqs | KV Util |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 9 | 4 | 28.98 | 565.4 | 51.33 | 32,673 | 0 | 0.16 |
| 10 | 8 | 36.30 | 902.6 | 62.26 | 39,982 | 0 | 0.31 |
| 11 | 16 | 49.97 | 1311.4 | 77.20 | 54,602 | 0 | 0.62 |
| 12 | 24 | 61.16 | **1607.4** | 96.07 | 69,221 | **0** | **0.93** |
| 13 | 32 | 94.71 | **1384.0** | 101.79 | 97,465 | **7** | **0.97** |
| 14 | 48 | 151.41 | **1298.5** | 100.00 | 105,427 | **23** | **0.97** |

**The Anomaly:** Rather than scaling linearly, throughput **peaks at batch 24 (1607.4 tok/s)** and then **collapses** to 1384.0 tok/s at batch 32 (-13.9%) and 1298.5 tok/s at batch 48 (-19.2%), while request turnaround time blows up from 61.16s to 151.41s (+147%).

### 2. Explanation of the Mechanism
1. **Exceeding KV Cache Physical Limits:** As proven in B1, the GPU holds at most ~25 full 4096-token sequences.
2. **Preemption Eviction:** 
   - At batch 32, $32 - 25 = 7$ active sequences exceed the memory pool. The scheduler preempts exactly $7$ sequences (`preempted_seqs = 7`).
   - At batch 48, $48 - 25 = 23$ sequences cannot fit. The scheduler preempts exactly $23$ sequences (`preempted_seqs = 23`).
3. **Recomputation Thrashing:** In vLLM/PagedAttention architectures, preempted sequences have their KV blocks evicted. When scheduled again, the serving engine must **re-prefill** the entire 3584-token prompt from scratch (recomputation). This re-executes billions of FLOPs, wastes high-bandwidth memory, stalls the decode pipeline, and causes tail latency (`e2e_ms_p95`) to surge to 105.4 seconds.

### 3. Proposed Configuration Change & Predicted Quantitative Effect
- **Proposed Change:** Enforce **Admission Control / Concurrency Limiting** by configuring `max_num_seqs = 24` in the vLLM engine (or front-end queue).
- **Alternative Config Change:** Enable **FP8 KV Caching** (`--kv-cache-dtype fp8`).
- **Predicted Quantitative Effect:**
  - *With `max_num_seqs = 24`:* Requests beyond 24 wait in an external FIFO queue rather than thrashing VRAM. Processing 48 requests in two clean batches of 24 takes $2 \times 61.16\text{s} = 122.32\text{s}$ wall-clock time, compared to $151.41\text{s}$ with preemption thrashing—delivering a **19.2% wall-clock reduction**, restoring throughput from $1298.5\text{ tok/s}$ back to **$1607.4\text{ tok/s}$ (+23.8%)**, and eliminating all 23 preemptions.
  - *With FP8 KV Cache:* KV bytes per token drops from 112 KiB to 56 KiB. Maximum concurrent sequences doubles from ~25 to **~51 sequences**, allowing Batch 48 to execute entirely in memory with **0 preemptions**, raising throughput to **>2100 tok/s**.

---

## B3. The Misread Column & Honest Goodput

### 1. The Misread Column
The intern based both conclusions on `reported_tok_s`.
- **Why it is misleading:** `reported_tok_s` is a raw token-counter metric measuring:
  $$\text{reported\_tok\_s} = \frac{(\text{prompt\_len} + \text{gen\_len}) \times \text{num\_requests}}{\text{wall\_clock\_s}}$$
- In Row 11 (batch 16, prompt 3584, gen 512):
  $$\frac{(3584 + 512) \times 16}{49.97\text{ s}} = \frac{65{,}536}{49.97} = \mathbf{1311.5\text{ tok/s}}$$
- In Row 6 (batch 16, prompt 512, gen 256):
  $$\frac{(512 + 256) \times 16}{13.91\text{ s}} = \frac{12{,}288}{13.91} = \mathbf{883.4\text{ tok/s}}$$

The intern mistook this metric for **generation output rate (goodput)**. In long-prompt workloads, **87.5% of the tokens are prompt tokens** ($3584 / 4096$). Prompt processing (prefill) is compute-bound matrix multiplication operating at tens of TFLOPs, processing thousands of tokens in parallel per second. Generation (decode) is strictly memory-bandwidth bound, generating only one token per step. Counting prefill tokens inflates `reported_tok_s` without delivering any additional generated output to users.

### 2. Honest Goodput of the Batch-24 Long-Prompt Row

Row 12 parameters: `batch_size = 24`, `prompt_len = 3584`, `gen_len = 512`, `wall_clock_s = 61.16s`, `ttft_ms_p50 = 500.5ms`, `itl_ms_p50 = 96.07ms`. Total generated tokens = $24 \times 512 = \mathbf{12{,}288\text{ tokens}}$.

#### Derivation Way 1: End-to-End Generated Output per Wall-Clock Second
$$\text{Goodput}_{\text{Way 1}} = \frac{\text{Total Generated Tokens}}{\text{Wall Clock Time}} = \frac{24 \times 512}{61.16\text{ s}} = \frac{12{,}288}{61.16} = \mathbf{200.92\text{ gen tok/s}}$$

*(If isolating the pure decode phase by subtracting the median TTFT prefill time $0.500\text{s}$: $\frac{12{,}288}{61.16 - 0.500} = \frac{12{,}288}{60.66} = \mathbf{202.57\text{ gen tok/s}}$)*.

#### Derivation Way 2: From Inter-Token Latency (`itl_ms_p50`)
During autoregressive decoding, each step produces 1 token per concurrent stream in batch. With a median step time of $\text{ITL} = 96.07\text{ ms} = 0.09607\text{ s}$:
$$\text{Goodput}_{\text{Way 2}} = \frac{\text{Batch Size}}{\text{ITL}} = \frac{24\text{ tokens}}{0.09607\text{ s}} = \mathbf{249.82\text{ gen tok/s}}$$

*(Note: Way 1 accounts for tail delays and scheduler overhead across the entire 61-second run, while Way 2 reflects the instantaneous steady-state decode rate).*

### 3. What the Report Should Have Said
> *"Longer prompts do not improve serving throughput; they artificially inflate `reported_tok_s` because prefill tokens are processed in parallel at high FLOP/s. In reality, longer prompts consume 7× more KV cache per request ($401\text{ MB}$ vs $56\text{ MB}$), reducing maximum GPU concurrency from >64 to ~25. At batch 24, true generation goodput is only **~201 gen tok/s** (not 1607 tok/s). Furthermore, batch 48 will **not** deliver 3200 tok/s; it exceeds the KV cache limit, causing 23 sequence preemptions and a throughput collapse to 1298 tok/s."*

---

## B4. Serving Stack Verification Metric

To confirm the B2 preemption mechanism in production, pull the metric:

**`vllm:num_preemptions_total`** (counter) or **`vllm:iteration_tokens_recomputed_total`**

**Expected Value:**
- For batch sizes $B \le 24$: Exactly **0**.
- For batch size $B = 32$: An immediate spike showing **$\ge 7$** sequence preemptions.
- For batch size $B = 48$: A severe escalation showing **$\ge 23$** sequence preemptions with thousands of recomputed tokens.
