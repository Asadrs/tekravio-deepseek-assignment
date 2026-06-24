# DeepSeek V4 Pro — AI Research Intern Assignment
**Tekravio Academy · Assignment 04 · Submitted by: [R.S.Asaduddin]**

---
## Table of Contents

- Introduction
- Module 1: DeepSeek V4 Pro Architecture
- Module 2: Licensing Analysis
- Module 3: Benchmark Analysis
- Module 4: Cost & Deployment
- Module 5: Enterprise Adoption
- References

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

DeepSeek maintains two separate product lines serving different purposes:

- **V-series (V3, V4 Pro, V4 Flash):** General-purpose models. Handle coding, summarization, tool use, agentic workflows, creative writing, and conversation. Built on MoE architecture.
- **R-series (R1):** Dedicated reasoning models, optimised through reinforcement learning for extended chain-of-thought on math, logic, and scientific problems.

**How V4 Pro incorporates reasoning without being a pure reasoning model:**

V4 Pro includes a **selectable thinking mode** (Non-Think, Think High, Think Max) that allows the model to switch between fast direct responses and extended chain-of-thought reasoning depending on the task. This gives it reasoning capability without the latency overhead of always running in reasoning mode.

**When to choose R1 over V4 Pro:**
- Pure math olympiad-level problems or graduate-level physics
- Tasks where maximum reasoning depth matters more than cost or speed
- Existing R1 deployments with fine-tuned checkpoints
- Code execution-heavy tasks (R1 has a slight HumanEval edge)

**When V4 Pro is sufficient (or better):**
- Agentic workflows requiring tool use and function calling (V4 Pro has native support; R1 does not)
- Long-context tasks exceeding 128K tokens (V4 Pro supports 1M tokens; R1 is capped at ~164K)
- Production workloads where cost matters (R1 is ~335% cheaper per token, but V4 Pro's long-context capability makes it the only viable option for full-codebase ingestion)
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

Winning these competitions demonstrates:
- **True generalisation** — the ability to solve genuinely unseen problems
- **Mathematical creativity** — constructing novel proofs and algorithms, not retrieving memorized ones
- **Algorithmic problem decomposition** — breaking complex problems into solvable subproblems under time pressure

For the IMO, this requires advanced number theory, combinatorics, and geometry. For IOI, it requires efficient algorithm design and data structures. For ICPC, it requires speed, team strategy, and breadth across domains. These cannot be faked through benchmark overfitting.

This is stronger evidence of mathematical and algorithmic reasoning capability than any benchmark score, because the problems are guaranteed to be out-of-distribution.

---

*Sources: DeepSeek V4 technical documentation, vLLM blog (April 2026), DataCamp DeepSeek V4 vs GPT-5.5 analysis, LLMReference comparison data, Towards Data Science MLA deep dive, Tech Jacks Solutions V4 architecture breakdown.*

---

## Module 02 — The MIT License & What Open Weights Actually Enable (20 marks)

---

### 1. MIT vs Other Open Licenses

Not all "open" AI models are equally open. Here's how the major licenses compare:

| License | Used by | Commercial use | Fine-tune & redistribute | Train other models | Patent grant |
|---|---|---|---|---|---|
| **MIT** | DeepSeek V4 Pro, R1 | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No explicit grant |
| **Apache 2.0** | Llama 4 Scout, Qwen 3.x, Gemma 4 | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Explicit patent grant |
| **DeepSeek prior license** | V3 base, Coder-V2, VL2 | ⚠️ Restricted | ⚠️ Restricted | ❌ No | ❌ No |
| **Grok license** | xAI Grok models | ✅ Yes | ⚠️ Restricted | ❌ Prohibits using weights to train other models | ❌ No |
| **Meta Llama 3 license** | Llama 3 family | ⚠️ Restricted for large-scale commercial use | ⚠️ Some restrictions | ⚠️ Restrictions apply | ❌ No |

**Key distinctions:**

- **What MIT allows that Apache 2.0 also allows:** Both permit commercial deployment, product integration, fine-tuning, and redistribution without royalties. An enterprise can build and sell a SaaS product powered by DeepSeek V4 Pro under either license with no licensing fees.
- **What MIT allows that Grok's license prohibits:** Grok's license explicitly prohibits using the model weights to train other models. Under MIT, you can use DeepSeek V4 Pro outputs or weights to distill smaller models, apply RLHF, or create derivative models — and release them under any license, including proprietary ones. This is explicitly permitted by DeepSeek.
- **One practical difference between MIT and Apache 2.0:** Apache 2.0 includes an explicit patent grant, meaning contributors grant users a royalty-free patent license. MIT does not. For most teams this is a minor distinction; for patent-sensitive industries (pharmaceuticals, hardware), Apache 2.0 can be preferable.
- **Older DeepSeek models** used a split license — permissive code license but more restrictive weights license. V4 Pro applies the MIT license to both code and weights, eliminating this ambiguity.

The only requirement MIT imposes is including the original copyright notice in any distribution. That's it.

---

### 2. Self-Hosting Economics — Full Break-Even Calculation

**Hardware requirements for V4 Pro:**

DeepSeek V4 Pro weights are approximately **865GB** on disk in FP4+FP8 mixed precision. This has significant hardware implications:

| Precision | VRAM needed | Minimum hardware |
|---|---|---|
| FP4+FP8 (native) | ~865GB | 8× B300 (Blackwell, single node) |
| FP8 only (Hopper) | ~865GB | 2-node H200 cluster (16× H200) |
| FP8 on H100 | ~865GB | Does not fit on 8× H100 (640GB total) |

For practical self-hosting, **V4 Pro requires a minimum of two H200 nodes** to run at full 1M token context. A single 8× H100 node (the most commonly available GPU cluster) is not sufficient.

**Monthly rental cost (approximate, as of mid-2026):**

| Provider | Configuration | Monthly cost |
|---|---|---|
| Lambda Labs / CoreWeave | 8× H200 (1-year reserved) | ~$17,500/month |
| AWS p5.48xlarge (H100 ×8, on-demand) | 8× H100 | ~$39,600/month |
| AWS p5.48xlarge (1-year reserved) | 8× H100 | ~$23,800/month |
| RunPod (spot, H100 ×8) | 8× H100 | ~$8,000–12,000/month |

Note: V4 Pro at 865GB does not fit on 8× H100 (640GB), so the H100 rows above require either two nodes or running V4 Flash instead.

**Break-even calculation (V4 Pro, API vs self-host):**

DeepSeek V4 Pro standard API pricing: $1.74/M input tokens, $3.48/M output tokens.

Assuming a typical 70/30 input/output split:
- Blended API rate ≈ (0.7 × $1.74) + (0.3 × $3.48) = $1.218 + $1.044 = **$2.26/M blended tokens**

For self-hosting on 8× H200 at $17,500/month (reserved, Lambda Labs):
- To break even: $17,500 ÷ $2.26 per million = **~7.7 billion tokens/month needed**
- At 65% GPU utilisation with ~5K tokens/second aggregate throughput: max capacity ≈ ~9.7B tokens/month
- Break-even volume: approximately **250M tokens/day of sustained, consistent load**

**Conclusion:** Self-hosting V4 Pro wins on cost only if you are processing more than ~250 million tokens per day at steady-state utilisation. Below that, DeepSeek's API is cheaper. Most startups never reach this volume. The primary reasons to self-host are **data sovereignty and compliance**, not cost.

---

### 3. Fine-Tuning Feasibility

**Full-parameter fine-tuning:**

Full-parameter fine-tuning of a 680B–1.6T parameter MoE model requires storing gradients, optimiser states, and activations alongside the weights. For a model this size, this requires an impractical number of GPUs — likely hundreds of H100s — and is not feasible for any team below hyperscaler scale. This is not a realistic option for most organisations.

**LoRA / QLoRA (practical option):**

QLoRA (Quantized Low-Rank Adaptation) is the practical fine-tuning approach. It works by:
- Quantizing the base model to 4-bit (dramatically reducing memory)
- Training only small low-rank adapter matrices (~0.1–1% of total parameters)
- Keeping the base model frozen

For V4 Pro via QLoRA: estimated 24–48 hours of fine-tuning on 8× H100 for a domain adaptation task. Together AI announced hosted fine-tuning for V4 Pro in April 2026 at $50/run base cost plus token cost.

**What a 10-person engineering team can realistically do:**
- **QLoRA domain adaptation:** Feasible via Together AI's hosted fine-tuning or a rented GPU cluster for a targeted task (legal documents, medical terminology, proprietary code style)
- **Full fine-tune:** Not feasible without enterprise GPU infrastructure at scale
- **Inference serving:** V4 Flash (284B, ~158GB) is the practical self-hosting target for small teams; V4 Pro requires dedicated GPU cluster operations

---

### 4. Geopolitical Procurement Considerations

**EU enterprise customers:**

DeepSeek is a Chinese company (Hangzhou DeepSeek Artificial Intelligence Basic Technology Research Co., Ltd.). This creates real regulatory risk for EU enterprises.

Using the **DeepSeek API** means user data is transmitted to and stored on servers in China. Italy's Garante blocked DeepSeek's service in January 2025 after finding it failed to meet GDPR obligations — specifically that personal data was stored in China with no adequacy decision in place under GDPR Chapter V, and no valid legal basis for processing EU residents' data.

**Does self-hosting the MIT weights change this?**

Partially, yes. If an EU company self-hosts the weights on EU-based servers:
- No data is sent to DeepSeek's infrastructure
- The GDPR data transfer concern (data going to China) is eliminated
- The company becomes the data controller and must comply with GDPR in their own right

However, GDPR obligations still apply:
- The company must have a valid lawful basis for any personal data processed through the model
- They must provide transparency to data subjects
- They remain liable for the model's outputs affecting EU residents
- The EU AI Act (taking full effect August 2026) still applies — they must classify their AI system by risk tier and meet the corresponding requirements

**US government contractors:**

The US government has not enacted a blanket federal ban on DeepSeek as of June 2026, but:
- FedRAMP authorization is required for cloud services used by federal agencies — DeepSeek's API has no FedRAMP authorization
- Multiple US government agencies and the US Congress have banned DeepSeek on government devices
- US government contractors working on sensitive or classified programmes should treat DeepSeek API usage as prohibited
- Self-hosting on US-based, cleared infrastructure reduces (but does not eliminate) procurement risk, as the model's Chinese origin remains a concern for national security use cases

---

### 5. Impact on the AI Industry

The MIT license on a frontier-adjacent model is a significant competitive event. Its effects:

**Pricing pressure on closed providers:** When DeepSeek V3 launched in late 2024 and V4 in April 2026, it demonstrably compressed the pricing of closed-weight providers. The intelligence-per-dollar ratio that V4 Pro offers ($1.74/M input vs GPT-5.5 at $5/M or Claude Fable 5 at $10/M) forces closed labs to either lower prices or justify their premium through capabilities DeepSeek cannot match (multimodal input, computer use, guaranteed safety documentation, enterprise SLAs).

**Open-weight model competition accelerating:** The open-weight frontier has compressed rapidly. The 2026 Stanford AI Index documents that the top closed model leads the top open model by only 3.3% on the Arena Leaderboard — a gap that was far larger two years earlier. DeepSeek's releases have forced other labs (Alibaba's Qwen, Google's Gemma, Meta's Llama) to also release more capable open-weight models.

**Ecosystem development:** MIT licensing creates maximum incentive for the developer ecosystem to build on V4 Pro. Fine-tune providers (Together AI), inference hosts (Fireworks, OpenRouter), and integration layers (Claude Code, OpenCode, CodeBuddy) all adopted V4 on launch day — something that would not happen as quickly under a restrictive license.

**Vendor lock-in reduction:** For any company that has built products on closed API-only providers, the availability of frontier-adjacent open weights creates a credible alternative and migration path. This shifts negotiating leverage in the market.

---

*Sources: DeepSeek V4 Hugging Face model card (MIT License), Verdent AI self-hosting guide, andrew.ooo cost analysis (April 2026), IAPP DeepSeek regulatory analysis, EU AI Act regulatory guidance, Italy Garante DeepSeek ban documentation, Lushbinary self-hosting guide.*

---

