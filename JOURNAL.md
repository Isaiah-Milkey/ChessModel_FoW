# Project journal

One entry per experiment or decision, newest at the bottom. Write it as you go, not at the end.
This is where the evidence for your grade lives (see docs/GRADING.md).

## Template

### YYYY-MM-DD: short title
- **Question / hypothesis:** what you expect and why.
- **Commit:** the commit hash the experiment ran at, and what you changed.
- **Setup:** data, model size, training compute (GPU-hours), games played for evaluation.
- **Result:** numbers with intervals (for example `fow.arena` ratings), plus anything you saw in games.
- **Decision:** what you will do next, and why.
- **AI help:** what an assistant did for this entry, if anything.

---

### 2026-10-24: reproduced the baseline
*(Drafted by Claude Code from the run output; reviewed by Isaiah.)*
- **Question / hypothesis:** Does the shipped baseline (`agents/baseline`) reproduce its published local rating (1217) on my own machine?
- **Commit:** `64c4157` (unmodified template).
- **Setup:** Local Windows machine, RTX 3060 Ti, Python 3.12, PyTorch 2.6 (CUDA), Stockfish 19
  (Windows x86-64 build, since `scripts/get_stockfish.sh` only fetches the Linux build).
  No training, evaluation only. A 4-game-per-opponent smoke test first (4 s), then:
  `python -m fow.arena agents/baseline bot:greedy bot:guess-sf-4x500 --games 100 --device cuda --workers 4 --out runs/baseline_100.jsonl`
  (200 games total, seed 0, about 12 s wall-clock).
- **Result:**
  | Opponent | Score |
  |---|---|
  | bot:greedy | 77.0 / 100 |
  | bot:guess-sf-4x500 | 78.0 / 100 |

  Baseline local rating **1246** (95% interval 1176–1316), consistent with the published 1217.
  All 200 games ended by king capture. The baseline scored about the same against both bots
  even though guess-sf-4x500 is rated 60 points higher, which is within noise at 100 games.
  Evaluation is CPU/Stockfish-bound; the GPU barely matters for a 314K-parameter network.
- **Decision:** This is just a test with setting up the baseline, next I will most likely scale up the model to test if that improves performance.
- **AI help:** Claude Code installed dependencies, downloaded Stockfish, ran the arena and drafted this entry.
