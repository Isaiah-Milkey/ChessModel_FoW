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

### 2026-11-06: 19x more data, and training until plateau
*(Facts drafted by Claude Code from the run output; content written by Isaiah.)*
- **Question / hypothesis:** How much can we scale up training to improve accuracy?
- **Commit:** models trained on `7139047` plus the uncommitted `--save-best` flag; the flag is
  `3ffb86c` and the best model is `27c5996` (`agents/baseline_75k`).
- **Setup:**
  - Data: `generate_data.py --games 100000 --workers 12 --out data/imitation_100k --seed 2`.
    Workers 5, 6 and 9 died silently at about 25 and 40 minutes in (no error in the log or in
    Windows event logs), and the run hung waiting for them, so I stopped it. The 9 finished
    shards hold 1,520,478 positions from about 75,000 games (3.7x the 20K set, 19x the baseline).
  - Training: 64x4 for 6 epochs, then 64x4 and 128x4 for 16 epochs with `--save-best` (keeps
    the epoch with the best validation match). RTX 3060 Ti: 44 s per epoch for 64x4, 90 s for 128x4.
  - Evaluation: 200 games each against bot:greedy and bot:guess-sf-4x500, plus 400 games head to head.
- **Result:**
  | Model | Epochs (best) | Validation match | vs greedy | vs guess-sf-4x500 | Rating (95%) |
  |---|---|---:|---:|---:|---|
  | 64x4 | 6 (6) | 42.2% | 168 / 200 | 167 / 200 | 1317 [1284, 1360] |
  | 64x4 (`baseline_75k`) | 16 (12) | 42.1% | 176 / 200 | 168 / 200 | 1347 [1298, 1402] |
  | 128x4 | 16 (7) | 42.4% | 163 / 200 | 166 / 200 | 1298 [1248, 1346] |

  - 64x4 plateaued: validation stayed at 42.1% from epoch 9 to 12, then slipped to 41.9%.
  - 128x4 peaked at epoch 7, then overfit: validation fell to 40.0% by epoch 16 while training
    loss fell from 1.83 to 1.36.
  - Head to head, 64x4 (epoch 12) beat 128x4 (epoch 7) 216 / 400 (54%, 365 unique games),
    at the edge of noise. The 64x4 6-epoch model vs the 20K-data 64x4: 208 / 400 (52%).
  - Compared with the baseline (1205), `baseline_75k` is about +140 Elo and plays at the
    same size and compute (4e7 FLOPs per move).
- **What surprised me:** The bigger model didn't help. This tells me that the size model we have now is good enough with the data I am currently using. (Will have to change architecture or data next to improve accuracy). We are limited by stockfish training.
- **Decision:** Move forward with this as my model/architecure.
- **AI help:** Claude Code ran the data generation, diagnosed and stopped the hung run, added
  `--save-best`, trained and evaluated the models, wrote the model card and drafted the facts
  in this entry.

### 2026-11-06: fading memory (Idea 1)
*(Drafted by Claude Code from the run output; edited by Isaiah.)*
- **Question / hypothesis:** The baseline's memory marks every enemy piece it has seen at full strength until that square is seen again, so a sighting from 20 turns ago looks as reliable as one from last turn. If old sightings fade (lose strength each turn), the network can tell fresh information from stale guesses and should play better.
- **Commit:** run on `3b7ee59` plus the uncommitted `--agent-dir` flag in
  `baselines/imitation/generate_data.py` and the uncommitted `agents/memory_age/`.
- **Setup:**
  - Memory (`agents/memory_age/memory.py`): the same three rules as `fow.memory.Memory`, plus
    the ply each remembered piece was last seen. Age = own turns since it was seen.
  - Encoding (`features.py`): memory planes 13-18 hold `max(1, round(255 * DECAY ** age))`
    instead of 1. With DECAY 0.85 a sighting keeps 52% of its strength after 4 turns; with
    0.6 it keeps 13%. The model divides these planes by 255, so the network is unchanged.
  - Data: the generator now takes `--agent-dir` to encode positions with an agent's own
    features. Without the flag the output is byte-identical to before, and the 11 tests pass.
    I replayed the exact games, labels and train/validation split of the earlier runs (only
    the memory planes differ):
    - 20K: `--games 20000 --workers 12 --seed 1` (405,888 positions).
    - 75K: `--games 100000 --workers 12 --seed 2`, keeping the 9 shards `baseline_75k` used
      (1,520,478 positions). Part of shard 7's labels came out different from the original
      run (23,498 labels, from 63.8% of the way through), so I used the original labels there.
    - DECAY 0.6: re-encoded the 0.85 data (the stored value tells you the age), 39 s, no new games.
  - Training: 64x4 with the same settings as the model it is compared with (20K: 8 epochs;
    75K: 16 epochs with `--save-best`, seed 0), about 12 min per 75K run on the RTX 3060 Ti.
  - Evaluation: 200 games each against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000
    on fresh arena seeds, the same seed for every model compared, plus head-to-head games.
- **Result:**
  | Model | Fade | Data | Epoch (best) | Validation match | Seed | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) |
  |---|---|---|---|---:|---|---:|---:|---:|---|
  | 64x4 | none | 20K | 8 | 40.2% | 3 | 162 | 162 | 136 | 1315 [1279, 1353] |
  | 64x4 | 0.85 | 20K | 8 | 40.3% | 3 | 173 | 162 | 135 | 1333 [1298, 1372] |
  | `baseline_75k` | none | 75K | 12 | 42.1% | 4 | 173 | 160 | 140 | 1338 [1296, 1372] |
  | 64x4 | 0.85 | 75K | 10 | 42.0% | 4 | 175 | 175 | 148 | 1388 [1359, 1423] |
  | `baseline_75k` | none | 75K | 12 | 42.1% | 5 | 170 | 165 | 139 | 1340 [1306, 1369] |
  | 64x4 | 0.85 | 75K | 10 | 42.0% | 5 | 174 | 167 | 139 | 1351 [1318, 1385] |
  | `baseline_75k` | none | 75K | 12 | 42.1% | 6 | 165 | 170 | 143 | 1348 [1310, 1374] |
  | 64x4 | 0.85 | 75K | 10 | 42.0% | 6 | 169 | 172 | 142 | 1357 [1326, 1392] |
  | 64x4 | 0.6 | 75K | 12 | 42.0% | 6 | 173 | 173 | 140 | 1363 [1320, 1396] |

  Scores are out of 200.
  - **Against the bots, fading leans slightly better:** +18 (20K), then +50, +11 and +9 Elo
    on three seeds at 75K. On seeds 4 and 5 combined, fade 0.85 scored 978 / 1200 against
    the baseline's 947. Every interval overlaps.
  - **Head to head, it is a tie:** fade 0.85 vs `baseline_75k` 510.5 / 1000 (51.1%, noise is
    about ±3%). At 20K, 217 / 400 (54%).
  - **The fade rate does not matter:** fade 0.6 vs fade 0.85 490 / 1000 (49%), and 1363 vs 1357
    on the same seed.
  - **Validation match did not move:** 42.0-42.1% for all three 75K models.
- **What surprised me:** I was surprised the memory did not change but I belive it will be useful in the future.
- **Decision:** Keep and test in future models.
- **AI help:** Claude Code wrote the fading memory and the `--agent-dir` flag, regenerated
  and re-encoded the data, found and fixed the shard 7 label mismatch, trained and evaluated
  the models and drafted the facts in this entry.
