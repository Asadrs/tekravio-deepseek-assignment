# DeepSeek V4 Pro — AI Research Intern Assignment
**Tekravio Academy · Assignment 04 · Submitted by: R.S.Asaduddin**

## Table of Contents
- Module 1: DeepSeek V4 Pro Architecture
- Module 2: Licensing Analysis
- Module 3: Benchmark Analysis
- Module 4: Cost & Deployment
- Module 5: Enterprise Adoption
- References

---

## Module 01 — MLA Architecture & the 88% Inference Cost Reduction (22 marks)

---

### 1. MoE with 680B Total / 42B Active Parameters

**What is Mixture-of-Experts (MoE)?**

A standard AI model (called a "dense" model) activates every single one of its parameters for every token it processes. If it has 42 billion parameters, it uses all 42 billion for every word — that's computationally expensive.

DeepSeek V4 Pro uses a different approach called **Mixture-of-Experts (MoE)**. Think of it like a hospital with many specialist doctors. When a patient arrives, a receptionist (the "router") sends them to only the relevant specialist — not every doctor in the building.

DeepSeek V4 Pro has **680 billion total parameters** spread across many "expert" sub-networks. For each token, a routing mechanism activates only **~42 billion parameters** — the most relevant experts for that specific input. The router is a lightweight gating network trained alongside the model; it scores each expert for relevance and selects the top-k experts per token.

**Inference latency implications:**
- A *dense* 42B model activates all 42B parameters per token — high compute, predictable memory usage.
- A *MoE* 680B model activates only 42B per token — same per-token compute cost as the dense 42B model, but with far greater total capacity and knowledge depth.
- The tradeoff is that all 680B parameters must still fit in GPU memory (or be sharded across GPUs), even though most are idle at any given moment.

**Why doesn't MoE at 680B require 680B's worth of compute per token?**

Because compute cost scales with *active* parameters, not total parameters. MoE decouples model capacity (how much the model knows) from inference cost (how much compute is used per token). You get the intelligence of a 680B model at the inference cost of a ~42B model.

---

### 2. Multi-head Latent Attention (MLA) Explained

**The problem MLA solves: the KV Cache bottleneck**

When an AI generates text token by token, it needs to "remember" everything it has processed so far. This memory is stored in what's called the **KV Cache** (Key-Value Cache). In standard Multi-Head Attention (MHA), every token in the context has its own full set of Key and Value vectors stored in GPU memory. The longer the conversation, the more memory the cache consumes — it grows linearly with context length.

For a 1 million token context window (roughly 750,000 words), standard MHA would require enormous GPU memory just for the cache. This makes long-context inference expensive or impossible.

**How different attention mechanisms compare:**

| Mechanism | How it works | Memory usage | Used by |
|---|---|---|---|
| **MHA** (Multi-Head Attention) | Full Key/Value vectors per head per token | High | GPT-3, early transformers |
| **GQA** (Grouped Query Attention) | Multiple Query heads share fewer KV heads | Medium | Llama, Mistral, Qwen |
| **MLA** (Multi-head Latent Attention) | Compresses KV into a small latent vector, decompresses at inference | Low | DeepSeek V2, V3 |
| **CSA/HCA** (V4 hybrid) | Reduces *number* of KV entries along the sequence dimension | Very Low | DeepSeek V4 Pro |

**How MLA works — in plain English:**

Instead of storing full-size Key and Value matrices for every token, MLA compresses them through a low-rank bottleneck:

1. **Compression step:** The input for each token is projected down into a small "latent" vector (much smaller than the original KV vectors). Think of it like saving a compressed thumbnail instead of a full-resolution photo.
2. **Decompression step:** When attention needs to be computed, the latent vector is projected back up to reconstruct the approximate Key and Value matrices.

This means the KV cache stores tiny latent vectors instead of full KV matrices — dramatically reducing memory. The approximation is close enough that model quality is preserved.

**DeepSeek V4 Pro goes further:** V4 Pro uses a new **hybrid attention** architecture — Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) — that reduces the *number* of KV entries that need to be stored across long sequences, not just the size of each entry. At a 1 million token context, V4 Pro's per-token inference FLOPs are approximately 27% of DeepSeek V3.2, and the KV Cache footprint is approximately 10% of V3.2. The 88% inference cost reduction figure refers primarily to this KV cache pressure reduction vs standard attention at long contexts.

---

### 3. Native Thinking Integrated with Tool Use

DeepSeek V4 Pro integrates **reasoning and tool-use decisions within the same thinking chain**. When the model needs to call an external tool (like a web search or code executor), it reasons about *which* tool to call, *what arguments* to pass, and *what to do with the result* — all as part of its internal thinking before producing output.

**How this differs from GPT-5.5's function calling:**

GPT-5.5 uses a structured function-calling mechanism where the model outputs a tool call in a predefined JSON schema format, waits for a result, then continues generation. The reasoning about what tool to use is largely implicit — it happens in the model's forward pass but is not made visible or integrated into a visible chain-of-thought.

V4 Pro's approach allows the model to *explicitly reason* about multi-step tool use before committing. For example: "I need to call the database tool first, check whether the result is empty, and if so, fall back to the web search tool." This conditional reasoning happens before any tool call is made.

**Quality improvement for multi-step tool tasks:**

In benchmarks involving complex multi-step tasks (e.g., financial research requiring multiple API calls), DeepSeek V4 Pro with thinking enabled reaches strong scores on analytical depth tasks — particularly game theory, competitive mapping, and multi-company comparison workflows. The integrated reasoning reduces errors from premature tool calls and incorrect argument construction.

---

### 4. V4 Pro vs R1 vs R2

**DeepSeek's model families:**

- **V-series (V3, V4 Pro, V4 Flash):** General-purpose models. Handle coding, summarization, tool use, agentic workflows, creative writing, and conversation. Built on MoE architecture.
- **R-series (R1):** Dedicated reasoning models, optimised through reinforcement learning for extended chain-of-thought on math, logic, and scientific problems.

**How V4 Pro incorporates reasoning without being a pure reasoning model:**

V4 Pro includes a **selectable thinking mode** (Non-Think, Think High, Think Max) that allows the model to switch between fast direct responses and extended chain-of-thought reasoning depending on the task.

**When to choose R1 over V4 Pro:**
- Pure math olympiad-level problems or graduate-level physics
- Tasks where maximum reasoning depth matters more than cost or speed
- Code execution-heavy tasks (R1 has a slight HumanEval edge)

**When V4 Pro is sufficient (or better):**
- Agentic workflows requiring tool use and function calling (V4 Pro has native support; R1 does not)
- Long-context tasks exceeding 128K tokens (V4 Pro supports 1M tokens; R1 is capped at ~164K)
- General-purpose coding, RAG, and document processing

**R2 status:** As of June 2026, DeepSeek R2 has not been officially released. Any claims about R2 should be treated as unverified until DeepSeek publishes an official model card.

---

### 5. V3.2-Speciale Competition Performance

DeepSeek V3.2-Speciale won gold at the **International Mathematical Olympiad (IMO)**, **International Olympiad in Informatics (IOI)**, and **International Collegiate Programming Contest (ICPC) 2026** — making it the first AI to achieve this triple.

**What competition wins demonstrate that benchmark scores cannot:**

Standard benchmarks like MATH-500 or HumanEval test problems that have appeared in training data, or problems structurally similar to training data. A model can score 97% on MATH-500 partially through pattern recognition on familiar problem types.

IMO, IOI, and ICPC problems are:
- **Novel:** Problem setters actively design problems that have never appeared anywhere before, specifically to prevent memorization.
- **Compositional:** Solutions require combining multiple mathematical or algorithmic concepts in non-obvious ways.
- **Verified under live conditions:** Problems are solved under time constraints with no access to references, and judged by human experts and automated test suites.

Winning these competitions demonstrates true generalisation, mathematical creativity, and algorithmic problem decomposition — none of which can be faked through benchmark overfitting.

*Sources: DeepSeek V4 technical documentation, vLLM blog (April 2026), DataCamp DeepSeek V4 vs GPT-5.5 analysis, LLMReference comparison data.*

---

## Module 02 — The MIT License & What Open Weights Actually Enable (20 marks)

---

### 1. MIT vs Other Open Licenses

Not all "open" AI models are equally open. Here is how the major licenses compare:

| License | Used by | Commercial use? | Fine-tune allowed? | Train other models? | Restrictions |
|---|---|---|---|---|---|
| **MIT** | DeepSeek V4 Pro, GLM-5.2 | ✅ Yes | ✅ Yes | ✅ Yes | Include copyright notice only |
| **Apache 2.0** | Llama 4 Scout, Qwen 3.x, Mistral | ✅ Yes | ✅ Yes | ✅ Yes | Include copyright + patent notice |
| **DeepSeek prior license** | DeepSeek V2, V3 (earlier) | ✅ Yes | ✅ Yes | ✅ Yes | Slightly more restrictive ToS clauses |
| **Grok 2.5 license** | xAI Grok 2.5 | ✅ Yes | ✅ Yes | ❌ No | Prohibits using weights to train other models |
| **Meta Llama 3 license** | Llama 3 family | ✅ Yes (under 700M MAU) | ✅ Yes | Restricted | >700M MAU requires Meta permission |

**Key differences:**

MIT and Apache 2.0 both permit commercial use, modification, and redistribution — and for most practical enterprise purposes they are equivalent. The difference is that Apache 2.0 includes an explicit **patent grant** (protecting users from patent claims by contributors), while MIT does not.

What MIT allows that **Grok 2.5's license prohibits**: you can take DeepSeek V4 Pro's weights and use them to train, fine-tune, or distill entirely new models — including commercial ones — and release those models under any license you choose, including proprietary ones.

---

### 2. Self-Hosting Economics (Full Calculation)

**Hardware requirements:**

| Model | Precision | Weight size | Minimum VRAM | GPU configuration |
|---|---|---|---|---|
| V4-Flash | FP4+FP8 (native) | ~158 GB | ~175 GB (with KV cache) | 2× H200 or 4× A100 80GB |
| V4-Flash | FP8 (H100-compatible) | ~284 GB | ~320 GB | 4× H100 80GB |
| V4-Pro | FP4+FP8 (native) | ~862 GB | ~1.1 TB | 8× H200 (single node) |
| V4-Pro | FP8 (H100) | ~2.4 TB | ~2.4 TB | 16× H100 80GB (multi-node) |

**Note:** MoE memory is total, not active. All 862 GB of V4-Pro weights must fit in VRAM because the router selects different experts per token.

**Monthly rental cost (V4-Pro, June 2026 estimates):**
- 8× H200 on Lambda Labs / CoreWeave (reserved, 1-year): ~$17,500/month
- 16× H100 cluster (on-demand): ~$25,000–$35,000/month

**Break-even calculation vs DeepSeek API ($1.74/$3.48 standard):**

Assume 70/30 input/output token ratio:
- Blended API cost ≈ ($1.74 × 0.7) + ($3.48 × 0.3) = **~$2.26 per million blended tokens**
- To break even on 8× H200 at $17,500/month: $17,500 ÷ $2.26 = **~7.7 billion tokens/month needed**

**Conclusion:** Self-host V4-Pro for **sovereignty, compliance, or fine-tuning reasons** — not for raw cost savings at moderate volumes. Most startups never reach the volume needed to justify it on cost alone.

---

### 3. Fine-Tuning Feasibility

**Full-parameter fine-tuning (not practical for most teams):** Full-parameter fine-tuning of a 680B+ MoE model requires compute equivalent to or exceeding the original training run — billions of GPU-hours. Not feasible outside well-funded AI labs.

**LoRA / QLoRA (practical for a 10-person engineering team):** QLoRA reduces fine-tuning memory requirements dramatically by training only small adapter layers on top of a quantized frozen base. For V4-Pro, QLoRA fine-tuning on domain-specific data is achievable in approximately 24–48 hours on an 8× H100 setup. Together AI has also announced hosted fine-tuning for V4-Pro at approximately $50 per run base cost plus token costs.

**Realistic customisation options for a small team:**
- QLoRA fine-tuning on proprietary datasets (domain adaptation)
- Prompt engineering and system-prompt customisation (no compute required)
- Distillation: using V4-Pro to generate training data for a smaller, faster model

---

### 4. Geopolitical Procurement Considerations

**EU enterprise customers:** DeepSeek's terms of service disclose that user data is stored on servers in China. Italy's Garante data protection authority banned the DeepSeek app and launched enforcement investigations. Germany's Berlin data protection commissioner declared DeepSeek's practices unlawful under GDPR. Multiple EU regulators — including France, Belgium, and Ireland — have opened formal investigations.

The core GDPR concern: China has no EU adequacy decision, meaning personal data transferred to China does not have equivalent legal protection. Under GDPR Articles 44–49, sending EU personal data to China requires either Standard Contractual Clauses (SCCs) or other appropriate safeguards — which DeepSeek has not been shown to provide.

**Does self-hosting change the risk calculation?** Yes — significantly. If an EU enterprise self-hosts V4 Pro's MIT weights on infrastructure within the EU, **no personal data is transmitted to DeepSeek or to China**. The GDPR data transfer problem disappears. The enterprise still carries obligations as the data controller, but the Chinese data residency risk is eliminated.

**US government contractors:** FedRAMP certification (required for US federal cloud services) is not available for DeepSeek's API, which routes through Chinese infrastructure. US government contractors must use FedRAMP-authorized alternatives (Together AI or Fireworks AI host V4 Pro on US infrastructure) or self-host the weights in a US-authorized environment.

---

### 5. Impact on the AI Industry

**Pricing pressure on closed providers:** When a model of comparable capability is available at $1.74/M input tokens with freely downloadable weights, closed-weight providers charging $10–$30/M for equivalent capability lose a primary argument for their pricing. OpenAI, Anthropic, and Google have all reduced API prices for comparable models since DeepSeek's open-weight releases began.

**Response from other AI labs:** Other labs have accelerated their own open-weight releases. Meta has open-sourced Llama 4 under Apache 2.0. Alibaba's Qwen 3 series is also Apache 2.0 licensed. GLM-5.2 (June 2026, MIT licensed) is now competing at the top of open-weight intelligence rankings.

The competitive dynamic has fundamentally shifted: **the question is no longer whether open weights can match closed models, but by how much closed models still lead and for how long.**

*Sources: DeepSeek V4 Hugging Face model card (MIT License), Lushbinary self-hosting guide (April 2026), AI Mastery vLLM deployment analysis, Pinsent Masons EU AI Act analysis, IAPP DeepSeek regulatory analysis, MindStudio DeepSeek V4 enterprise review.*

---

## Module 03 — Benchmark Analysis (20 marks)

---

### 1. What Each Benchmark Actually Tests

Before comparing scores, it's important to understand what each benchmark measures — and what it doesn't.

| Benchmark | What it tests | Why it matters | Limitation |
|---|---|---|---|
| **SWE-bench Verified** | Resolving real GitHub issues — navigate a real codebase, write a patch that passes real tests | Closest proxy to actual software engineering work | Vendor-reported scores vary; needs independent reproduction |
| **SWE-bench Pro** | Same as Verified but uses issues filed *after* training cutoff | Reduces data contamination risk | Harder to game; scores are lower across all models |
| **LiveCodeBench** | Code generation on freshly-released competitive programming problems | Mitigates training-set contamination | Favours algorithmic thinking over real-world engineering |
| **GPQA Diamond** | Graduate-level science questions (PhD-level physics, chemistry, biology) | Tests deep domain reasoning | Now saturating at the frontier; small gaps are noise |
| **MMLU-Pro** | 57-subject multiple choice knowledge test | Good for comparing general capability floor | Saturating — 87–90%+ scores are hard to differentiate |
| **HLE (Humanity's Last Exam)** | Extremely hard, expert-authored questions across all domains | Designed specifically to not be saturated | Only useful as a relative comparison, not absolute |
| **Terminal-Bench 2.0** | Autonomous CLI tasks — file systems, build tools, shell commands | Best proxy for agentic system tasks | GPT-5.5 optimised specifically for this |
| **Codeforces Rating** | Competitive programming rating based on real contest performance | Live, contamination-resistant | Algorithmic, not real-world engineering |

---

### 2. Full Benchmark Comparison Table

All scores are vendor-reported unless noted. Treat as directional — independent reproduction is still landing for some V4 Pro numbers.

| Benchmark | DeepSeek V4 Pro | Claude Opus 4.7 | GPT-5.5 | Winner |
|---|---|---|---|---|
| **SWE-bench Verified** | 80.6% | 87.6% | 74.9% | Claude Opus 4.7 |
| **SWE-bench Pro** | 55.4% | 64.3% | 58.6% | Claude Opus 4.7 |
| **LiveCodeBench** | 93.5% | 88.8% | ~85% | **DeepSeek V4 Pro** |
| **Codeforces Rating** | 3,206 | ~2,800 | 3,168 | **DeepSeek V4 Pro** |
| **GPQA Diamond** | 90.1% | 94.2% | 93.6% | Claude Opus 4.7 |
| **MMLU-Pro** | 87.5% | 89.1% | 88.1% | Claude Opus 4.7 |
| **HLE (no tools)** | 37.7% | ~40% | 41.4% | GPT-5.5 |
| **Terminal-Bench 2.0** | 67.9% | 65.4% | 82.7% | GPT-5.5 |
| **MATH-500** | 96.1% | ~95% | ~95% | **DeepSeek V4 Pro** |
| **Context window** | 1M tokens | 200K tokens | 1.05M tokens | Tie (V4 Pro / GPT-5.5) |
| **Output price** | $0.87/M | $25/M | $30/M | **DeepSeek V4 Pro** |

**Key takeaway from the table:**
- V4 Pro **wins** on: LiveCodeBench, Codeforces, MATH-500, and cost by a wide margin
- Claude Opus 4.7 **wins** on: SWE-bench (both), GPQA Diamond, MMLU-Pro — i.e. real-world software engineering and knowledge recall
- GPT-5.5 **wins** on: Terminal-Bench (autonomous CLI tasks), HLE, long-context retrieval

---

### 3. Verifying DeepSeek's Capability Claims

DeepSeek's own release notes describe V4 Pro as:
> *"Leading all current open models in math, STEM, and coding, while trailing current proprietary models."*

This is an honest and accurate self-assessment. Let's verify each claim:

**"Leading all open models in coding"** — Verified. SWE-bench Verified at 80.6% is the highest confirmed score among open-weight models. V4-Pro-Max leads the open-weight SWE-bench Verified leaderboard, tied with Gemini 3.1 Pro. LiveCodeBench at 93.5% is also the highest open-weight result.

**"Trailing proprietary models"** — Verified. Claude Opus 4.7 leads SWE-bench Verified (87.6%) and SWE-bench Pro (64.3%) by meaningful margins. GPT-5.5 leads Terminal-Bench 2.0 (82.7%) and HLE (41.4%). The gaps are real but not generational.

**The 3–6 month gap claim:** DeepSeek's own technical report states V4 Pro trails the absolute frontier by roughly 3–6 months. Based on benchmark evidence this is plausible — V4 Pro sits between GPT-5.4 and GPT-5.5 on most evals, suggesting it corresponds to approximately where the proprietary frontier was in late 2025.

**Important caveat:** Several V4 Pro scores are still vendor-reported as of June 2026 and await independent reproduction. The SWE-bench Verified score in particular (80.6%) has not yet been confirmed by Scale AI's standardized SEAL leaderboard. Independent trackers report slightly different numbers (80.6–83.7% across different scaffolding setups). Until Scale SEAL publishes an entry, treat V4 Pro's SWE-bench score as directionally accurate but not finalized.

---

### 4. What Competition Wins Mean as Enterprise Evidence

DeepSeek V3.2-Speciale won gold at IMO, IOI, and ICPC 2026 — competitions where:
- Problems are novel and designed specifically to prevent memorization
- Solutions are verified against automated test suites and human judges
- Time pressure and breadth requirements prevent narrow overfitting

**Why this matters for enterprise evaluation:**

Standard benchmarks can be gamed through training data contamination, prompt engineering for specific benchmarks, or cherry-picking evaluation conditions. A model that scores 97% on MATH-500 might still fail on a novel problem type it has never seen.

Competition wins provide contamination-resistant evidence because the problems are literally designed to have never appeared before. A model winning IMO gold demonstrates genuine mathematical reasoning — not pattern matching on memorized proofs. For an enterprise evaluating whether to trust a model with novel analytical tasks, this is stronger evidence than any benchmark score.

**Practical implication:** If DeepSeek V4 Pro can approach IMO-gold-level mathematical reasoning, tasks like financial modelling, scientific data analysis, and complex algorithm design are well within its reliable capability range. These are the tasks where V4 Pro's cost advantage makes the biggest business impact.

---

### 5. Where V4 Pro Wins and Where It Loses — Honest Assessment

**V4 Pro clearly wins:**
- Competitive and algorithmic programming (Codeforces 3,206 — best among all models)
- Code generation on fresh problems (LiveCodeBench 93.5%)
- Mathematical reasoning (MATH-500 96.1%)
- Cost efficiency — 30x cheaper output than Claude Opus 4.7
- Long-context tasks (1M token window with efficient CSA/HCA attention)
- Open-weight availability — only option for air-gapped / data-sovereign deployment

**Claude Opus 4.7 clearly wins:**
- Real-world software engineering (SWE-bench Pro 64.3% vs V4 Pro's 55.4% — a genuine 9-point gap)
- Multi-file repository-level code changes
- Writing quality, citation rigor, structured long-form output
- Production agentic loops with complex tool-use chains
- Multimodal input (V4 Pro has no vision capability)

**GPT-5.5 clearly wins:**
- Autonomous CLI and shell-command workflows (Terminal-Bench 82.7%)
- Computer use and GUI automation (OSWorld-Verified 78.7%)
- Long-context retrieval (MRCRv2 87.5% at 128–256K)

**The honest summary:** V4 Pro is not "better" or "worse" than Claude or GPT-5.5 — it is differently strong. For cost-sensitive coding, math, and algorithmic tasks with open-weight requirements, it is the best choice. For production software engineering agents and writing-heavy work, Claude Opus 4.7 still leads. For autonomous system tasks and multimodal work, GPT-5.5 leads.

*Sources: Lushbinary frontier model comparison (April 2026), DataCamp V4 vs GPT-5.5 analysis, BenchLM.ai V4 Pro benchmark profile, Codersera V4 Pro review (updated June 2026), SWE-bench Pro leaderboard (morphllm.com, June 2026), FundaAI 38-task benchmark study.*

---

## Module 04 — Pricing & Deployment Strategy (18 marks)

---

### 1. The Full Pricing Picture

DeepSeek V4 Pro launched April 24, 2026 at $1.74/M input and $3.48/M output. On May 22, 2026, DeepSeek made a permanent 75% price cut:

| Token type | Original price | Permanent price |
|---|---|---|
| Input (cache miss) | $1.74/M | **$0.435/M** |
| Input (cache hit) | $0.0145/M | **$0.003625/M** |
| Output | $3.48/M | **$0.87/M** |

**Full model pricing comparison (June 2026):**

| Model | Input ($/M) | Output ($/M) | Context | Open weights? |
|---|---|---|---|---|
| **DeepSeek V4 Flash** | $0.14 | $0.28 | 1M | ✅ MIT |
| **DeepSeek V4 Pro** | $0.435 | $0.87 | 1M | ✅ MIT |
| **Claude Sonnet 4.6** | $3.00 | $15.00 | 200K | ❌ |
| **Claude Opus 4.7** | $5.00 | $25.00 | 200K | ❌ |
| **GPT-5.5** | $5.00 | $30.00 | 1.05M | ❌ |
| **Gemini 3.1 Pro** | $3.50 | $10.50 | 1M | ❌ |

V4 Pro output is **34x cheaper than GPT-5.5** and **29x cheaper than Claude Opus 4.7**.

---

### 2. Intelligence-Per-Dollar Comparison

Raw price means little without quality context. The real question is: how much intelligence per dollar?

**Cost to run 1 million output tokens:**
- DeepSeek V4 Pro: **$0.87**
- Claude Opus 4.7: **$25.00** — 28.7x more expensive
- GPT-5.5: **$30.00** — 34.5x more expensive

Claude Opus 4.7 leads V4 Pro on SWE-bench Verified by ~7 points (87.6% vs 80.6%). To get the same number of successful outputs, you'd need to run V4 Pro ~28x more volume for the same cost. In most workloads, the cost gap dwarfs the quality gap.

**The cache hit multiplier — the number most teams miss:**

DeepSeek cache hit pricing is $0.003625/M tokens — roughly 120x cheaper than GPT-5.5's input price. For agent workflows where system prompts and tool definitions repeat across thousands of calls, cache hits dominate the bill.

Concrete example:
- 100,000 chat turns/day with a 6,000-token system prompt
- **Without cache:** 100,000 × 6,200 tokens × $0.435/M = **$269.70/day**
- **With cache hits:** ~**$10.88/day**
- **Saving: 96% reduction** just from prompt caching

Cache-aware system design is the single biggest cost lever for V4 Pro deployments.

---

### 3. DeepSeek's Pricing Strategy — What It Signals

**Why DeepSeek can price this aggressively:**
- MoE architecture — only 49B parameters activate per token, so compute cost is proportional to active parameters not total
- CSA/HCA attention reduces KV cache memory by 90% vs V3.2 at 1M context — hardware costs per token are dramatically lower
- Chinese infrastructure costs are substantially lower than US equivalents
- DeepSeek is private and not under quarterly revenue pressure — they can price for market share

**What making the 75% cut permanent means:**

A promotional discount is a marketing event. A permanent cut is a market floor. DeepSeek published a reference price every other lab must respond to. A model at $0.87/M output delivering 80%+ SWE-bench Verified performance forces OpenAI, Anthropic, and Google to either match pricing or defend premium pricing through genuinely superior capability.

**DeepSeek's likely long-term playbook:** Win developer mindshare and infrastructure adoption now. Once V4 Pro is embedded in production stacks, switching costs rise — enabling future platform monetisation. The same playbook AWS used with S3 in 2006.

---

### 4. Hybrid Deployment Architecture

The right architecture is intelligent routing — not using one model for everything.

**Routing rules by task type:**

| Task type | Recommended model | Reason |
|---|---|---|
| High-volume chat, summarisation, extraction | V4 Flash ($0.14/$0.28) | 107x cheaper than GPT-5.5, sufficient quality |
| Competitive programming, algorithm design | V4 Pro | Best LiveCodeBench (93.5%) and Codeforces (3,206) |
| Mathematical reasoning, STEM analysis | V4 Pro | Best MATH-500 (96.1%), strong IMO performance |
| Multi-file GitHub issue resolution | Claude Opus 4.7 | 87.6% SWE-bench — 7 points ahead of V4 Pro |
| Autonomous CLI / shell-command agents | GPT-5.5 | 82.7% Terminal-Bench — clearly leads |
| Multimodal input (images, documents) | Claude Opus 4.7 or GPT-5.5 | V4 Pro has no vision capability |
| Air-gapped / data-sovereign deployment | V4 Pro self-hosted | Only frontier-adjacent MIT-licensed open option |
| Long-context tasks (>200K tokens) | V4 Pro or GPT-5.5 | Only models with verified 1M token support |

**Expected cost saving from routing vs using Claude for everything:**

For a typical mixed workload (60% simple, 30% coding, 10% complex agentic):
- All Claude Opus 4.7: ~$25/M output blended
- Routed (V4 Flash + V4 Pro + Claude): ~$3–5/M blended
- **Estimated saving: 80–88% reduction in inference cost**

---

### 5. When to Upgrade from API to Self-Hosted

Use the API by default. Self-hosting makes sense only when:

- **Data sovereignty is non-negotiable** — Healthcare (HIPAA), regulated finance, EU GDPR-strict, US government contractors. Self-hosting on your own infrastructure eliminates Chinese data residency risk. Cost comparison becomes irrelevant.
- **You need custom fine-tuning** — The hosted API cannot serve fine-tuned checkpoints. Domain adaptation (legal, medical, proprietary code style) requires self-hosting.
- **You have existing GPU capacity to amortise** — If H100s/H200s are already owned for training, running V4 Flash inference on spare capacity has near-zero marginal cost.
- **Volume exceeds ~250M tokens/day at sustained utilisation** — Only at this scale does dedicated hardware at reserved pricing begin to beat API costs.

For everyone else: use the API. At $0.87/M output with zero infrastructure overhead, the economics are hard to beat.

*Sources: DeepSeek official API pricing page (api-docs.deepseek.com), Apidog V4 Pro permanent price cut analysis (May 2026), Codersera V4 Pro pricing guide (May 2026), Enterprise DNA pricing announcement, CloudZero DeepSeek pricing guide, OpenRouter V4 Pro listing.*

---

## Module 05 — Enterprise Risk Framework (20 marks)

*Coming soon*
