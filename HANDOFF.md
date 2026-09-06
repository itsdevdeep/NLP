# Handoff — PS4 Conversational AI Assignment

## What this is
BITS AIML ZG521 assignment: build & compare LLM-only, RAG, and Agentic AI assistants for a
university-support domain (PS4). Deliverable: one executed `.ipynb`,
`PS4_GroupN_AIMLZG521_ConversationalAI_Assignment.ipynb`, naming needs `GroupN` → real group
number before submission. Header also has placeholder names/IDs to fill in.

## Environment
- Real Python 3.14.7 installed (was previously just a Windows Store stub). Venv at `.venv/`.
- Ollama running locally, model `qwen3:14b-q4_K_M`. `gemma4:12b` also pulled but unused so far.
  **Nothing in this notebook works without a reachable Ollama server with this model.**
- **GPU: AMD Radeon RX 6700 XT, 12 GB (ROCm, not CUDA — no `nvidia-smi` on this box).** The
  6700 XT (gfx1031) is not officially ROCm-supported but works. Model runs **100% on GPU** only
  after `num_ctx` is pinned: qwen3's default context is 40960, which loads the model at ~16 GB
  (weights ~9 GB + KV cache ~6 GB) and spills ~35% to CPU. `llm_call` now sets `num_ctx=8192`
  (`NUM_CTX` in `make_notebook.py`) → ~10 GB, fully on GPU, ~39 tok/s (was ~10 tok/s split).
  Check with `ollama ps` — want `100% GPU`. If it ever shows a CPU split again, lower `NUM_CTX`.
  Nothing in the notebook needs more than ~3k tokens of context so 8192 is already generous.
- Stack: langchain 1.4.0 (+ community/text-splitters/huggingface), faiss-cpu, sentence-transformers,
  pypdf, pandas, ollama python client, jupyter/nbconvert. All confirmed working on Python 3.14.

## Corpus (Part A)
- 24 real PDFs from 8 US universities, downloaded to `data/raw/` (13MB). Manifest: `data/sources.tsv`.
- Extracted/cleaned to `data/processed/*.txt` + `_manifest.json` via **`build/extract.py`** —
  **this is a standalone script, run once manually, NOT called by the notebook.** The notebook only
  reads the already-processed output. If corpus needs to change, re-run this script by hand.
- Found and fixed real corruption during extraction: broken ligature font maps in 2 PDFs
  (`informaĕon`→`information` etc. in doc 21; decorative bullet glyphs in doc 15). Caught by
  auditing non-ASCII character counts across the corpus — worth doing this check on any new corpus.
- 318 pages, ~93K words, 1,110 chunks after `RecursiveCharacterTextSplitter` (800 char / 120
  overlap / 40-char floor to drop empty page-marker fragments).

## Notebook structure so far
Parts A–D plus Part E sections 4.5–4.6 (see "Part E so far" below). 71 cells, executes clean
A–E in ~10 min on GPU, 0 errors, no widget metadata.
- **Part A** — domain/users/intents, KB source table, preprocessing writeup, chunking (executed,
  real counts), preprocessing-harm justification, KB limitations, 5 conversation flow designs.
- **Part B** — LLM-only baseline. Two system prompts (basic vs structured) run head-to-head,
  `temperature=0` for reproducibility. **Key learning:** don't write the comparative analysis
  before executing — first draft assumed the basic prompt would hallucinate confidently, but
  `qwen3:14b` already hedges well even under a bare prompt. Rewrote B.5 to report the real,
  reproducible difference instead (structured prompt leads with a clarifying question, ~2x more
  concise, ~2x lower latency as a side effect — not a hallucination-suppression story).
- **Part C** — Conversational RAG. FAISS + `all-MiniLM-L6-v2`, cosine similarity, top-K=4,
  retrieval latency measured (~6-8ms). Query rewriting for follow-up turns, 5 multi-turn
  conversations (2 turns each) mapped to Part A's intents. **Found a real, reproducible bug, kept
  it in the notebook rather than hiding it:** the query rewriter sometimes drops the topic entity
  on short follow-ups ("What if I choose the thesis option instead?"), so retrieval drifts to the
  wrong document and the model cites the wrong source. Documented with the exact turns in C.5.
- **Part D** — Agentic AI / tool calling. LLM router → `{action, reasoning, tool, tool_args,
  rag_query, clarify_question}` JSON (temp 0, `think=False`); actions `direct / rag / tool /
  tool+rag / clarify / refuse`. Mock DB **Northgate University** built inline (10 courses, 9 exam
  slots — synthetic, disjoint from the corpus). Two tools `course_lookup` / `exam_schedule_lookup`
  over pandas frames. One 8-turn demo conversation hits every branch incl. clarify→resolution and
  RAG+tool. D.6 has a per-turn "why this action" table. **Learnings:**
  (1) `qwen3:14b` as router ignores a plain "never ask for the term" instruction and re-clarifies
  on resolvable follow-ups — added a small deterministic `_repair()` pass that rewrites a
  `clarify` to `course_lookup` when the *current* message names one unambiguous catalog code.
  Scan only the current message, not history — scanning prior turns pulls a stale code from 2
  turns back and hijacks a genuine clarify.
  (2) For `tool+rag`, retrieving on the raw bundled message drifts (pulled OSU TA-guidelines doc
  23 instead of the missed-exam syllabus doc 14); the router now emits a separate policy-only
  `rag_query` and retrieval uses that.
  (3) rag demo queries must be phrased like the corpus — "process when accused of misconduct"
  retrieves doc 06 reliably; "how many appeal attempts before dismissal" returns not-found.

## Decisions made (user-approved)
- Local Ollama only, $0 API cost, latency/compute reported instead of $ cost (Part F, not yet built).
- `think=False` for all Ollama calls — benchmarked: thinking mode is a ~7x latency tax with worse
  output on this model (truncates before answering). Router in Part D will use an explicit
  `reasoning` JSON field instead of hidden thinking.
- Doc #9 (a fragmentary slide deck) kept in the KB with a chunk-length floor, not dropped.
- Mock tool DB for Part D is a **synthetic, deliberately disjoint** university dataset
  ("Northgate University", invented courses/timetables/exam slots), not derived from the real
  corpus — keeps tool-vs-RAG routing boundary clean for evaluation. Built inline in the notebook
  (Part D.1), not a file under `data/mock_db/` (that dir is still empty and unused).
- Everything self-contained in the notebook — no external `.py` modules imported at grading time
  (only `build/extract.py` and `build/make_notebook.py` exist outside it, both pre-processing /
  authoring tools, not runtime dependencies).

## Running on another machine (teammate execution)
- `requirements.txt` at repo root = the notebook's pip deps (pinned). Not just Ollama — the RAG
  stack (langchain*, faiss-cpu, sentence-transformers→torch, transformers), pandas/numpy/
  matplotlib, and jupyter/nbconvert to execute it.
- Outside pip: install Ollama, then `ollama pull qwen3:14b-q4_K_M`. `gemma4:12b` not needed.
- First run needs internet once: downloads `all-MiniLM-L6-v2` (~90 MB) to `~/.cache/huggingface`.
- Ship the folder with `data/processed/` + `data/sources.tsv` (the notebook reads these).
  `data/raw/` (13 MB PDFs) and `build/` are only for re-processing the corpus — not needed to run.
- **macOS (M-series):** use Python 3.12 or 3.13, not 3.14 — torch/sentence-transformers arm64
  wheels are reliable there. Ollama uses Metal automatically; `num_ctx=8192` keeps the model
  ~10 GB, fine on 18 GB unified memory. No CPU/GPU config needed (the AMD/ROCm notes above are
  Windows-only).

## Build process (important, non-obvious)
The notebook is **generated from a script**, not hand-edited directly:
```
.venv/Scripts/python.exe build/make_notebook.py          # regenerates the .ipynb from scratch
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 PS4_GroupN_AIMLZG521_ConversationalAI_Assignment.ipynb
```
Edit `build/make_notebook.py`, regenerate, re-execute, re-verify outputs — don't hand-edit the
`.ipynb` JSON directly. Mistake made early on: forgot to update the hardcoded output filename in
the script after renaming the `.ipynb` by hand, so a regenerate silently wrote to the old filename
— check `NB_PATH` in the script matches the actual target file if renaming.

**Two file-divergence incidents — watch for this:**
1. Something (a Jupyter/VS Code tab open elsewhere, or a synced folder) overwrote the project
   `.ipynb` with a stale A–C-only copy mid-session. The generator was unaffected — recovered
   with one regenerate + execute. Keep only one editor on the file; if the project dir is in
   OneDrive/iCloud/Dropbox, pause sync while working.
2. User edited cells in a *downloaded* copy of the `.ipynb` (from the chat), not the project
   file — those edits never reached `build/make_notebook.py` and looked "missing". Always
   mirror any hand-edit into the generator; it is the only source of truth.

## Resolved issues
- **ipywidgets "Loading weights" bar** (VS Code *"Cannot read properties of undefined (reading
  'ipywidgetsKernel')"*): fixed. The plain-text-tqdm first cell works — after regenerate +
  execute the `.ipynb` has zero widget-mime outputs and no `metadata.widgets`. Verified
  programmatically; not eyeballed in the VS Code renderer but there is nothing left for it to
  choke on.
- **E.5 chart not rendering in the notebook UI** (saved to `results/` but no inline image):
  caused by `matplotlib.use("Agg")` in the E.5 cell — Agg is headless, emits no cell output.
  Removing the line alone isn't enough in a live kernel (`matplotlib.use()` is sticky per
  process — needs a kernel restart). Fixed for good: (1) `%matplotlib inline` as the first line
  of cell 1; (2) E.5 no longer touches the backend and ends with
  `Image("results/partE_comparison.png")`, which embeds the saved PNG into the cell output
  regardless of backend or kernel state. Verified: E.5 cell now carries an `image/png` output.

## Part E so far (sections 4.5 + 4.6 — done)
- **E.1** = 4.6 evaluation dataset: 25 queries, 7 categories (`EVAL_QUERIES`, inline). Each has
  `qid` / `category` / `turns` / `expect`. 5 multi-turn (2 turns each). `MULTI_TURN_QIDS` set.
- **E.2** runs all 3 systems (`run_llm_only` / `run_rag` / `run_agent`) over the shared set →
  `runs_df` (75 rows = 25 queries × 3 systems: answers, per-query wall-clock latency,
  retrieval/router trace). Cell ends with a `pivot_table` showing all 25 qids × 3 systems.
- **E.3** LLM-judge (`qwen3` judging `qwen3`) → JSON per (system, query): answer_quality,
  groundedness (1–5), context_accuracy (multi-turn only), hallucination (bool). Self-judging
  caveat is stated in the cell and E.6 — read category deltas, not absolute scores.
- **E.4** = 4.5 comparison table (`summary`): the required columns + a deterministic
  `groundedness_check` (cite-or-decline rate, independent of the judge) + per-category quality.
- **E.5** = 4.5 chart → `results/partE_comparison.png`, embedded inline via `Image(...)` (left:
  answer quality by the 7 categories — where the systems actually diverge; right: grounded-action
  + hallucination rates).
- **E.6** findings. **Real, executed results — do not "improve" these numbers:**
  - Grounded systems win on factual (direct-factual quality 2.6 → 4.4).
  - **RAG loses on ambiguous queries (2.0 vs baseline 5.0)** — retriever always returns a chunk,
    grounding prompt makes the model assert one program's number for a cross-doc-ambiguous
    question. Baseline asks which program; agent's `clarify` only partly recovers (3.0).
  - **RAG worst on prompt-injection (2.67 vs 5.0)** — `RAG_SYSTEM_PROMPT` has no refusal rule.
  - Hallucination ~0 everywhere; real error mode is mis-grounding (wrong `[doc_id]`).
  - **Agent is the *fastest* system** (mean latency orders Agent < RAG < LLM-only every run;
    absolute values drift run-to-run, ~1.7–2.2 / ~2.2–3.4 / ~4.8s) — refuse/clarify skip
    generation; grounded answers are shorter than the baseline's hedged paragraphs. E.6 no
    longer hardcodes latency; it reads from the `summary` table.
- Runtime: full A–E execute is ~10 min on GPU (was ~30+ CPU-split). ~200 Ollama calls in Part E.

## Not yet built
- **Part E cont.** — 4.7 metrics deep-dive (Hit Rate / P@K / R@K for RAG), 4.8 robustness
  stress-test table (≥5 domain tricky cases), 4.9 hallucination analysis (answer exists /
  partially / doesn't / false premise / OOD; does RAG reduce unsupported answers vs baseline).
  Much of the raw material is already in `runs_df` / `scores_df`.
- **Part F** — 250-400 word reflection: (a) high-traffic cost-sensitive, (b) low-volume
  high-stakes; must cite the group's own E results + name one limitation of the chosen arch.
  E.6 already points the answer: RAG for (a) on latency/cost but fix the ambiguous+injection
  gaps; agent for (b) on the refuse/clarify safety routes.
- Header still has placeholder group number / names / IDs — need real values before submission.

## Style note from user
Keep markdown cells terse — tables and short bullets, not prose paragraphs. Don't restate what
code output already shows. Avoid comments/scaffolding in code cells beyond what's necessary to
follow the logic. This was corrected once already (first two parts were over-written, trimmed
down after feedback) — hold this bar for Parts D-F from the start.
