# Ratiocinator Implementation Plan

## Approach: Vertical Slices, Local First

The system has deep dependency chains. Building horizontally (all of component A, then all of B) would delay feedback for months. Instead, build **thin vertical slices** that prove the core loop works end-to-end on a toy problem, then widen each layer.

The critical path is: **LLM interface → sandboxed execution → single experiment loop → tree search → distribution → synthesis.**

Literature RAG and paper generation are valuable but not on the critical path — they enhance quality, not core functionality.

---

## Phase 1: The Single Experiment Loop (Local)

**Goal:** An LLM proposes a code modification, it executes in a Docker container, metrics return to the orchestrator. One complete iteration.

### Steps

1. **Project scaffolding** — Python package structure, dependency management (pyproject.toml), basic CLI entry point.
2. **LLM interface via LiteLLM** — Unified client that can call any model. Config-driven model routing (coding vs. generalist). Start with one provider (Ollama or an API key you already have).
3. **Sandbox runner** — Given a git repo + branch, build a Docker image, run training to N steps, capture stdout/stderr and exit code. Return structured results (metrics dict + logs).
4. **Experiment loop** — Orchestrator prompts LLM with a task description + current code. LLM returns a code diff. Orchestrator applies the diff, runs the sandbox, collects metrics.

### Measurable Result
> Run `ratiocinator run --task "improve learning rate schedule" --repo ./example --steps 500` and get back a JSON with proposed change, execution logs, and metric values — without human intervention.

### Toy Problem
Use a minimal training script (e.g., MNIST classifier or a 2-layer MLP on synthetic data) as the target repo. Fast iteration, cheap to run, easy to verify.

---

## Phase 2: Tree Search (Local)

**Goal:** The orchestrator explores multiple branching code modifications, backtracks on failures, and converges on the best-scoring variant.

### Steps

5. **Tree data structure** — Nodes hold: parent ref, code diff, execution status, metric scores. Persistent to disk (SQLite or JSON) so runs survive crashes.
6. **Best-First Search algorithm** — Score function ranks nodes by metrics (e.g., lowest val_loss). Expansion: LLM generates N candidate modifications from the current best node. Selection: pick highest-scoring unexplored node.
7. **Error recovery** — On container crash/OOM, feed the stack trace + code back to the LLM and request a fix. This becomes a child node of the failed node.
8. **Search budget controls** — Max depth, max total nodes, max wall-clock time. Hard stops, not LLM-controlled.

### Measurable Result
> Run the tree search on the toy problem. After exploring ≥10 nodes across ≥3 depth levels, the best leaf outperforms the root's baseline metric. Visualize the tree as a simple graph (graphviz dot or terminal ASCII).

---

## Phase 3: Literature-Grounded Ideation

**Goal:** Hypotheses are informed by recent papers rather than pure LLM imagination.

### Steps

9. **arXiv retrieval module** — Query arXiv API by topic keywords. Download abstracts, cache locally. Basic semantic search over cached abstracts (sentence-transformers embeddings + FAISS or simple cosine similarity).
10. **Novelty filter** — Given a proposed hypothesis string, check similarity against retrieved literature. Flag near-duplicates. Prompt LLM to refine or pivot.
11. **Integration with tree search** — The ideation step at tree expansion queries the RAG pipeline. Each node stores its literature grounding (cited paper IDs + relevance scores).

### Measurable Result
> Given a research topic (e.g., "efficient attention mechanisms"), the system retrieves ≥5 relevant papers, generates ≥3 novel hypothesis candidates, and each candidate includes a citation to the paper that inspired it.

---

## Phase 4: Distributed Execution (Vast.ai)

**Goal:** Experiments run on ephemeral GPU instances in parallel. The orchestrator manages provisioning, monitoring, and teardown.

### Steps

12. **Vast.ai client wrapper** — Programmatic interface to `vastai` CLI: search offers, create instances, poll status, destroy instances. Typed Python wrapper with retries.
13. **Instance bootstrap** — `onstart.sh` template: clone repo at specific branch/commit, install deps, run training, push metrics via webhook, upload artifacts on completion.
14. **Webhook receiver** — Lightweight HTTP server on the orchestrator that receives metric pushes from running instances. Updates tree nodes in real time.
15. **Parallel tree search** — Expand multiple nodes simultaneously. Each expansion launches a Vast.ai instance. Throttle by configurable max-concurrent-instances.
16. **Safety controls** — `MAX_DOLLARS_PER_RUN` budget cap (query Vast.ai billing). Per-instance TTL enforcement. Kill orphaned instances on orchestrator shutdown.

### Measurable Result
> Launch a tree search that provisions ≥3 simultaneous Vast.ai instances on a real (cheap) GPU tier. All instances complete or are terminated within budget. Metrics from all instances appear in the tree. Total cost stays under $5.

---

## Phase 5: Synthesis and Paper Generation

**Goal:** The system produces a structured LaTeX report from experimental results.

### Steps

17. **Plotting agent** — Given a set of tree nodes with metrics, generate comparative plots (loss curves, ablation bar charts). matplotlib in a sandboxed subprocess.
18. **LaTeX template + generator** — Structured template (abstract, intro, method, results, conclusion). LLM fills each section given: hypothesis, code diffs, metrics, plots.
19. **Automated reviewer** — Second LLM pass critiques the draft on a rubric (clarity, supported claims, missing ablations). If score is below threshold, trigger revision. Max 3 revision cycles.
20. **Artifact publishing** — Upload final PDF, code diffs, checkpoints, and logs to HuggingFace Hub. Tag with git commit hash.

### Measurable Result
> After a completed tree search, the system outputs a PDF with: abstract, methodology section describing the code changes, results table, ≥2 plots, and a conclusion. The automated reviewer scores it ≥6/10 on its rubric.

---

## Dependency Graph

```
Phase 1 (Single Loop)
  └── Phase 2 (Tree Search)
        ├── Phase 3 (Literature RAG)  ← can start during Phase 2
        ├── Phase 4 (Distribution)
        │     └── Phase 5 (Synthesis)
        └── Phase 5 (Synthesis)       ← basic version possible after Phase 2
```

## First Concrete Step

Start with **Step 1 + 2**: project scaffolding and the LiteLLM wrapper. This is a single session of work and immediately testable:

```bash
# After step 2, you can verify:
python -m ratiocinator.llm "Suggest a learning rate schedule change for this code: ..."
# Should return a structured response from whatever model is configured.
```

Everything else builds on this foundation.
