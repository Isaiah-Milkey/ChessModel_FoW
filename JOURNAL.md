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

### 2026-11-05: scaling the model vs scaling the data
*(Drafted by Claude Code from the run output; to be reviewed by Isaiah.)*
- **Question / hypothesis:** Does a model with about 4x the parameters (128x4 instead of 64x4)
  beat the baseline? If not, is the limit the amount of training data?
- **Commit:** run on `57a356a` plus the uncommitted `agents/baseline_4x/`; committed afterwards.
- **Setup:** RTX 3060 Ti, Stockfish 19, every training run 8 epochs with seed 0.
  - Data: the baseline's data is not in the repo (`data/` is gitignored), so I regenerated it
    with `generate_data.py --games 4000 --workers 12 --seed 0` (81,059 positions, 170 s), then
    made a 5x set with `--games 20000 --workers 12 --seed 1` (405,888 positions, 819 s).
    The worker count changes which games are played.
  - Models: 64x4 (312,896 weights) and 128x4 (1,215,552 weights) trained on each dataset;
    each run took 35 s to 3.5 min.
  - Evaluation: 200 games each against bot:greedy and bot:guess-sf-4x500, plus 400 games
    of 128x4 vs 64x4 on the 406K data.
- **Result:**
  | Model | Data | Validation match | vs greedy | vs guess-sf-4x500 | Rating (95%) |
  |---|---|---:|---:|---:|---|
  | shipped baseline (64x4) | 83K | 39.4% (model card) | 158 / 200 | 134.5 / 200 | 1205 [1161, 1236] |
  | 64x4 | 81K | 36.8% | 152 / 200 | 138 / 200 | 1200 [1168, 1229] |
  | 128x4 | 81K | 36.4% | 154 / 200 | 144 / 200 | 1217 [1183, 1253] |
  | 64x4 | 406K | 40.3% (peak, epoch 6) | 165 / 200 | 159 / 200 | 1283 [1234, 1332] |
  | 128x4 | 406K | 41.0% (peak, epoch 5) | 162 / 200 | 164 / 200 | 1289 [1242, 1332] |

  - **Model size did not help.** On the same data the 128x4 model matched the 64x4 model.
    Head to head on the 406K data it scored 206.5 / 400 (51.6%), within noise. On 81K
    positions its training loss fell to 1.09 (64x4: 1.46) while validation did not improve,
    so the extra capacity went into memorizing.
  - **More data did help.** 5x the data raised validation match from about 37% to 40-41%
    and the rating by about 80 Elo over the baseline, for both sizes. More games do give new
    positions: 86% of the 81K positions were unique, and 97.9% of the unique positions in
    its second half were not in its first half.
  - **Training longer would not help at this size.** Both models on the 406K data peaked
    before epoch 8 and then overfit slightly.
  - My regenerated 81K data gave lower validation match than the shipped model card
    (36.8% vs 39.4%), possibly because the labels come from Stockfish 19 instead of 16.
- **Measurement notes:**
  - Networks that always play their top move are deterministic, so two of them playing each
    other give only 2 distinct games (one per color). My first such run (400 games) was
    meaningless; the 64x4 model came out at -141.
  - Fix in `agents/baseline_4x/agent.py`: the first 4 moves are sampled from
    `softmax(logits / 1.0)` using the game seed, then the top move. `fow.validate` still passes
    (same moves on replay), the 128x4 vs 64x4 games became 369 / 400 unique, and strength
    against the bots did not measurably change (1204 [1167, 1236] vs 1217 [1183, 1253]).
- **Decision:** data, not model size, is the bottleneck. Validation match may be nearing a
  ceiling around 40%, because some teacher moves depend on pieces the agent cannot see.
  Next: generate about 100K games and train 64x4 for 5-6 epochs to see whether the gain
  continues. Prefer 64x4 for the size-adjusted rating. If the gain flattens, move to Idea 1
  (memory) or self-play.
- **AI help:** Claude Code generated the data, trained and evaluated the models, found the
  repeated-games problem, wrote the random-opening change and drafted this entry.
