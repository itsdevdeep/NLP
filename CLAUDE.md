# PS4 Conversational AI Assignment — Session Log

See [HANDOFF.md](HANDOFF.md) for full project context (environment, corpus, notebook structure,
build process, decisions made). This file logs what changed in each work session.

## 2026-09-06/07 — Completed Parts E.7-F, opened PR

Picked up from HANDOFF.md's "Not yet built" list (Part E cont. + Part F) and finished the
notebook end-to-end.

- **Environment**: this session ran on a Mac (M-series), not the original Windows/AMD box.
  Created `.venv` with Python 3.13, installed `requirements.txt`, registered it as a Jupyter
  kernel (`nlp-venv`) — nbconvert must be run with `--ExecutePreprocessor.kernel_name=nlp-venv`
  or it silently falls back to a kernel missing the project's packages (`ModuleNotFoundError:
  langchain_text_splitters`). Ollama already had `qwen3:14b-q4_K_M` pulled; ran at 100% GPU
  (Metal) throughout, no CPU-split issues on this hardware.
- **E.7 (4.7 metrics)**: Hit Rate/Precision@K/Recall@K for RAG, computed against gold `doc_id`s
  parsed from the `expect` field of `direct_factual`/`multi_turn` queries (10 of 25 — the only
  categories with one documented source). Conversational-quality table (context retention, task
  completion, and a new deterministic `response_consistency` metric — citation-set overlap
  across multi-turn turns). Efficiency table (latency, token usage, retrieval latency) — required
  adding real token capture to `RAGSession`/`AgentSession` (previously only the LLM-only baseline
  tracked tokens).
- **E.8 (4.8 robustness)**: 8 stress cases reusing already-run E.2 answers — zero new LLM calls.
- **E.9 (4.9 hallucination)**: the 5 required cases (exists/partial/absent/false-premise/OOD) plus
  a category-level hallucination-rate table comparing RAG vs. baseline.
- **Part F**: 328-word reflection, written to cite the notebook's own tables by name rather than
  hardcode numbers, so it doesn't go stale on re-execution.
- **References section**: added after Part F — the notebook originally cited zero literature,
  which is a graded rubric line item (5%, "Research/literature component"). Five papers, each
  tied to a specific implementation choice (transformer backbone, the embedding model, the RAG
  retrieve-then-generate pattern, conversational query rewriting, the ReAct-style agent router).
- **Caught two stale claims** carried over from the original Windows/AMD execution that no longer
  matched this Mac's real numbers once re-executed here: retrieval latency (~163ms mean here, not
  "single-digit ms" — the embedder runs on CPU) and one quality figure (2.8 actual vs. 2.6
  written). Fixed both in the generator and mirrored into the executed notebook.
- **Header filled in**: Group 52; Ahamed Imthias (2025AE05985), Harsha Vardhan (2025AE05710),
  Vishvajith (2025AE05225), Devdeep Dasgupta (202505660). Notebook renamed
  `PS4_Group52_AIMLZG521_ConversationalAI_Assignment.ipynb` per the assignment's naming
  convention.
- **Final state**: 88 cells, 0 errors, full A-F execution. Verified `build/make_notebook.py`
  (source of truth) and the executed `.ipynb` are byte-identical on every markdown cell after
  every hand-sync — regenerate to a temp file and diff before trusting either one stale.

### Git/PR note
`origin` (`itsdevdeep/NLP`) rejects pushes from this machine's GitHub account (`ImthiasJ` has no
write access — 403). Worked around by forking to `ImthiasJ/NLP` (remote name `fork`), pushing the
`remainingSteps` branch there, and opening a cross-repo PR into `itsdevdeep/NLP:main`. If push
access gets granted later, push directly to `origin` instead of the fork.

### Still open
- Part A's chunk/count numbers, Part B/C/D demonstrations, and E's evaluation results are all
  real executed output as of this session — but re-running on different hardware/model builds
  will shift exact numbers again (as it did once already, moving from AMD/Windows to this Mac).
  If re-executed, diff the new numbers against the markdown prose (E.6, E.9, C.5) before trusting
  either — that prose makes specific numeric claims, not just qualitative ones.
