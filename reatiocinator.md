# Auto-Researcher: Distributed Scientific Discovery Pipeline

## Project Overview
The goal is to build a robust, open-source auto-researcher that bridges the gap between structured tournament optimization (like the `prx-tg` project) and the open-ended discovery of SakanaAI's AI-Scientist-v2. This system moves from an empirical hyperparameter optimization loop to an end-to-end scientific discovery pipeline.

While `prx-tg` serves as an excellent execution engine with strict compute budgets and step-based validation, the Auto-Researcher operates at a higher level: using Agentic Tree Search to propose novel hypotheses, edit source code, debug errors, plot results, and synthesize peer-reviewed-level LaTeX papers.

---

## Core Architectural Blueprint

### 1. Knowledge-Driven Ideation (The Literature RAG)
Instead of relying on human-defined changes or random exploration, the agent grounds its hypotheses in existing literature.
* **The Goal:** Before starting experiments, the agent reads recent literature to propose why a specific configuration or architectural change might work.
* **Implementation:**
    * **HuggingFace & arXiv APIs:** A Retrieval-Augmented Generation (RAG) pipeline using `huggingface_hub` and the `arxiv` Python library.
    * **Novelty Checking:** Queries APIs with proposed hypotheses (e.g., "Low-rank GaLore applied to DiT attention") to filter out existing ideas and pivot to novel variations.

### 2. Model Agnosticism
To ensure independence and openness, the system swaps proprietary APIs for a unified interface capable of running open-weight models.
* **The Goal:** Use a single interface for models like DeepSeek-Coder-V2, Qwen2.5, or Llama-3.
* **Implementation:**
    * **LiteLLM Abstraction:** Points to local inference endpoints (vLLM, Ollama) or API providers using identical code.
    * **Model Routing:** Routes coding tasks to coding-specific models (DeepSeek-Coder) and synthesis tasks to strong generalist models (Llama-3-70B).

### 3. Agentic Tree Search (Beyond YAML tweaks)
Moving beyond simple configuration changes, the system uses Best-First Tree Search (BFTS) to explore branching paths of actual code modifications.
* **The Goal:** Allow the agent to modify `model.py` or `train.py` while autonomously handling crashes and errors.
* **Implementation:**
    * **Sandboxing:** Wrap training loops in Docker containers to prevent host corruption from bad code.
    * **Tree Search Algorithm:** Maintains a tree of experiments. If a branch fails (e.g., OOM), it backtracks, analyzes the stack trace, and attempts a refined modification.
    * **Tournament Heuristics:** Uses the `prx-tg` tournament structure as the evaluation heuristic for the tree search (e.g., scoring by `reconstruction_lpips` at step 500).

### 4. Automated Synthesis and Write-up
Science requires narrative context beyond YAML outputs.
* **The Goal:** Generate comprehensive reports containing motivations, methodology, ablation studies, and plots.
* **Implementation:**
    * **Plotting Agent:** Access to a Python sandbox (matplotlib/seaborn) to parse logs and generate comparative graphs.
    * **LaTeX Generator:** Compiles plots, hypotheses, and empirical results into a structured LaTeX template.
    * **Automated Reviewer:** An "LLM-as-a-judge" critiques the draft and triggers revision cycles before final output.

### 5. Reproducibility & Open Science Engine
* **The Goal:** Automate the publishing pipeline so every successful experiment is easily verifiable.
* **Implementation:**
    * **Git Diffs:** Commits exact configurations and code diffs for every winning phase.
    * **HF Hub Integration:** Automatically uploads checkpoints, logs, and generated PDFs to Hugging Face repositories with associated git commit hashes.

---

## Leveraging Vast.ai for Distributed Research

Decoupling the "Brain" (LLM orchestrator) from the "Muscle" (distributed GPUs) shifts the system from a sequential optimizer to a massively parallel AI research lab, solving the "time-to-signal" bottleneck.

### 1. Parallel Agentic Tree Search
Instead of waiting for sequential experiments, the tree search becomes asynchronous and highly parallel.
* **Horizontal Exploration:** The agent generates multiple hypotheses simultaneously and spins up separate Vast.ai instances to race them.
* **Native Sandboxing:** Vast.ai instances are natively Docker containers. If an AI-written bug causes a crash, it is isolated. The orchestrator catches the exit code, feeds the trace back to the LLM, and provisions a new instance with a fix.

### 2. Hardware-Aware Research (Scaling Laws)
Research can include empirical hardware scaling and memory profiling across different GPU tiers.
* **Dynamic Provisioning:** The agent can request a cheap RTX 3090 for rapid exploration and automatically scale to an A100 (80GB) for high-depth testing.
* **Efficiency Optimization:** The agent can measure throughput (it/s) vs. hourly cost to include a "Cost-to-Train Analysis" in the final paper.

### 3. Distributed Infrastructure Pipeline
* **Git-Ops Sync:** The agent commits a new hypothesis to a branch; the orchestrator launches a Vast.ai instance with an `onstart` script that clones that branch and begins execution.
* **Telemetry:** The `autoresearch.py` script pushes metrics (val_loss, LPIPS, logs) back to the orchestrator via webhooks or tools like Weights & Biases (wandb).
* **Artifact Harvesting:** Winning checkpoints are uploaded to S3 or Hugging Face before the ephemeral instance is destroyed.

### 4. Safety & Circuit Breakers
Hard-coded constraints outside of LLM control prevent runaway costs.
* **Timeouts:** The orchestrator enforces a strict TTL (Time To Live). If a phase exceeds its expected duration, the instance is forcefully terminated.
* **Spend Budgets:** A `MAX_DOLLARS_PER_PAPER` limit is hardcoded in the Python orchestrator to prevent budget exhaustion from tree search loops.

---

## Final Architecture Summary

1.  **The Command Center (Local/VPS):** Runs LiteLLM, maintains the arXiv RAG database, manages the Best-First Tree Search state, and holds API keys.
2.  **The Dispatcher:** Commits code to GitHub and executes `vastai create instance` with automated `onstart` training scripts.
3.  **The Fleet (Vast.ai):** Ephemeral GPUs that pull code, train to the step-budget, upload artifacts, and signal completion.
4.  **The Compiler:** Converges on a global optimum, pulls final data, generates plots, and compiles the LaTeX paper.
