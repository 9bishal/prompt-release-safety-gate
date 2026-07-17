# Prompt Release Safety Gate

A CI gate that treats prompts like production code. Before a new prompt
version merges, it's automatically compared against the current production
prompt on a golden test set — catching regressions in **cost, length,
latency, and factual faithfulness** before they ship, the same way a failing
unit test blocks a bad code change.

**The motivating scenario:** someone edits a support-bot prompt to sound
"friendlier." It reads well in a quick manual check. In production it turns
out every response is 3x longer, costs 3x more in tokens, and — because the
new wording buried a required policy detail under paragraphs of warmth —
quietly gives wrong answers on refund edge cases. This gate is built to
catch exactly that, automatically, on the PR.

The repo ships with a working example of precisely this: `prompts/old_prompt.txt`
is a concise policy-bot prompt, `prompts/new_prompt.txt` is a "friendlier"
rewrite. Run the gate on both and watch it fail.

---

## How the advanced-RAG techniques do real work here

This isn't just a diff script with an LLM call bolted on — the same
hybrid-search / rerank / semantic-cache stack from RAG systems is used for
genuine reasons inside the gate:

| Technique | Role in the gate |
|---|---|
| **Hybrid search (BM25 + vector, RRF-fused) over the golden dataset** | As your test bank grows to hundreds of cases, running all of them on every PR is slow and expensive. The gate diffs the old vs. new prompt text and hybrid-searches the golden dataset for the cases most relevant to what actually changed — e.g. an edit to refund language pulls refund test cases first. |
| **Cross-encoder reranking** | Refines the hybrid-search candidates into a precise "smoke test" tier, plus always includes a non-negotiable "core" tier (out-of-scope/refusal behavior) regardless of relevance ranking. |
| **Semantic result cache** | Repeated CI runs on a lightly-amended PR would otherwise re-score every case from scratch. The cache embeds (prompt_text, question) pairs and skips re-running Groq entirely on a near-duplicate that's already been scored. |
| **Strict, grounded LLM-judge prompting** | The judge scoring faithfulness is instructed to grade only against the golden reference, forced to output structured JSON, and required to score 0.0 if the answer contradicts the reference or fails to escalate an out-of-scope question — so a "friendlier but wrong" answer can't pass on tone alone. |

---

## Architecture

```
   old_prompt.txt, new_prompt.txt
              |
              v
     diff_prompts()  -->  what actually changed?
              |
              v
   hybrid search + rerank golden_dataset --> relevant "smoke" cases
   (+ always-run core tier: out-of-scope refusal checks)
              |
              v
   for each selected case:
     check semantic cache --hit--> reuse cached result
     (miss) run old prompt + new prompt against Groq
     judge new output vs. golden reference (strict, grounded)
              |
              v
   aggregate metrics: length / cost / latency, old vs new
   paired significance test (Wilcoxon / paired t-test)
              |
              v
   apply thresholds --> PASS / WARN / FAIL
   render markdown report
              |
              v
   posted as a PR comment; job fails the build on FAIL
```

This is implemented as a plain orchestrator (`src/gate.py`) rather than a
LangGraph state machine, since the flow here is linear per-case rather than
branching — but it reuses every RAG building block from the sibling project
(hybrid retriever, reranker, semantic cache) with the same interfaces.

---

## Project structure

```
prompt-release-safety-gate/
├── prompts/
│   ├── old_prompt.txt          # current production prompt (concise, grounded)
│   └── new_prompt.txt          # candidate rewrite ("friendlier" -- demonstrates the regression)
├── golden_dataset/
│   └── cases.json              # versioned test cases: question + expected_reference + category
├── src/
│   ├── config.py                # all tunable thresholds, read from .env
│   ├── test_bank_retriever.py    # hybrid search + rerank over the golden dataset
│   ├── cache.py                  # semantic result cache (skips re-scoring unchanged pairs)
│   ├── runner.py                 # runs a prompt against Groq, captures real token usage
│   ├── judge.py                  # strict, grounded LLM-judge (faithfulness scoring)
│   ├── stats.py                  # paired significance testing (Wilcoxon / t-test)
│   ├── gate.py                   # orchestrates everything into a verdict
│   └── report.py                 # renders the verdict as a markdown PR comment
├── scripts/
│   └── run_gate_cli.py           # CLI entrypoint, used locally and in CI
├── .github/workflows/
│   └── prompt-gate.yml           # runs the gate on any PR touching prompts/ or golden_dataset/
├── tests/
│   └── test_gate_components.py   # offline tests (no API key needed)
└── reports/                      # gate_report.md / gate_report.json written per run
```

---

## Setup

```bash
git clone https://github.com/9bishal/prompt-release-safety-gate.git
cd prompt-release-safety-gate
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your free Groq API key from https://console.groq.com/keys
```

Run the gate locally:

```bash
python scripts/run_gate_cli.py --old prompts/old_prompt.txt --new prompts/new_prompt.txt
```

With the sample prompts included, this should **FAIL** — the "friendlier"
prompt produces noticeably longer, more expensive responses, caught as a
statistically significant regression. Open `reports/gate_report.md` to see
the full breakdown per test case.

To try a version that should **PASS**, edit `new_prompt.txt` down to a
change that doesn't blow up length/cost (e.g. adjust just one word), rerun,
and compare.

## Wiring it into CI

`.github/workflows/prompt-gate.yml` runs automatically on any pull request
that touches `prompts/` or `golden_dataset/`. Add `GROQ_API_KEY` as a repo
secret (Settings -> Secrets and variables -> Actions), and the gate will post
its markdown report as a PR comment and fail the check on a FAIL verdict --
exactly like a failing unit test blocking a merge.

## Tuning thresholds

All thresholds live in `.env` (see `.env.example`):

- `MAX_LENGTH_INCREASE_PCT` -- response length regression ceiling
- `MAX_COST_INCREASE_PCT` -- token cost regression ceiling
- `MAX_LATENCY_INCREASE_PCT` -- latency regression ceiling (WARN, not FAIL, by default)
- `MIN_JUDGE_SCORE` -- minimum faithfulness score any single case may score (FAIL below this)
- `SIGNIFICANCE_ALPHA` -- p-value cutoff for treating a metric delta as real signal vs. noise

## Design notes / trade-offs

- **Statistical significance, not raw thresholds alone**, gates the
  length/cost/latency checks -- a single noisy case shouldn't fail a build.
  The faithfulness (judge) check is a hard floor per-case instead, since one
  wrong answer to a customer is not something you average away.
- **Embeddings and reranking run locally** for the same reason as the RAG
  project -- retrieval/rerank quality shouldn't add to the token bill.
- **The result cache is a flat JSON file**, same storage-agnostic interface
  as before -- swap for Redis to share across CI runners/branches without
  touching callers.
- **The golden dataset is intentionally small (9 cases)** to keep this
  runnable in minutes on a free Groq tier. The retrieval/rerank layer is
  what lets this scale to hundreds of cases without every PR taking longer.

## Possible extensions

- Add a canary/shadow-traffic step: route a small % of real traffic to the
  new prompt and compare live metrics before full cutover.
- Track gate history in a small SQLite/Postgres table so you can bisect
  which prompt edit introduced a regression across many PRs.
- Swap the LLM-judge for RAGAS-style automated metrics (faithfulness,
  answer relevancy) if the prompts under test are themselves RAG prompts.

## License

MIT
# prompt-release-safety-gate
