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

### 2026-11-06: turn number and own last 3 moves (move_history)
*(Drafted by Claude Code from the run output; edited by Isaiah.)*
- **Question / hypothesis:** The network only sees the current turn, so it cannot tell the
  opening from the middlegame except through the pieces on the board, and it does not know what
  it was just trying to do. If it is told the turn number and its own last 3 moves, it should
  learn what to do at each stage of the game and play more consistent plans.
- **Commit:** `56dcb56` (`agents/move_history/` code, `--record` in
  `baselines/imitation/generate_data.py`, `baselines/imitation/reencode.py`); the trained
  weights and model card are new.
- **Setup:**
  - Encoding (`agents/move_history/features.py`, `history.py`): the baseline's 22 planes
    unchanged, plus plane 22 = ply number (capped at 100, scaled to 0..1 in `model.py`) and
    planes 23-28 = from and to squares of the agent's own last 3 moves, flipped for Black.
    318,089 parameters (`baseline_75k`: 314,057); only the first layer is bigger.
  - Data: 75,000 new games, seed 3, 1,514,142 positions (`baseline_75k` had 1,520,478),
    generated with `--record` so the games and labels were saved (50 min, 12 workers). The
    saved games were re-encoded with the baseline encoding in 44 s, with no Stockfish, as a
    control. Planes 0-21, labels and legal masks are identical in the two datasets.
  - What the new planes contain: median ply 21 (about move 10), 0.6% of positions at the
    ply cap; 95% / 90% / 85% of positions have a 1st / 2nd / 3rd previous move.
  - Training: 64x4, 16 epochs, `--save-best`, seed 0, the same for both; 11 min each on the
    RTX 3060 Ti.
  - Evaluation: 200 games each against bot:greedy, bot:guess-sf-4x500 and
    bot:guess-sf-16x1000 on arena seed 6 (the same seed as `baseline_75k` and the fade models),
    plus 1,000 games head to head against the control.
- **Result:**
  | Model | Planes | Epoch (best) | Validation match | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) |
  |---|---:|---|---:|---:|---:|---:|---|
  | `move_history` | 29 | 9 | 42.0% | 173 | 173 | 135 | 1353 [1322, 1387] |
  | control (`runs/base75k_s3`) | 22 | 9 | 42.0% | 169 | 165 | 135 | 1331 [1299, 1365] |
  | `baseline_75k` (earlier, other data) | 22 | 12 | 42.1% | 165 | 170 | 143 | 1348 [1310, 1374] |

  Scores are out of 200.
  - **Validation match did not move:** 42.0% for both at their best epoch. `move_history`
    reached a lower training loss (1.741 vs 1.764 at epoch 16) and dropped less after its
    peak (41.7% vs 41.4% at epoch 16), but its best was the same.
  - **In games it leans better:** +22 Elo against the bots (intervals overlap), and 526.5 / 1000
    head to head against the control (+18 Elo, 95% interval [-5, +35], p about 0.09). Not
    significant.
  - **The control rated 17 below `baseline_75k`** with the same code and the same amount of
    data, so a different set of training games alone moves the rating by about as much as
    the new planes did.
- **What surprised me:** I believe the model is overfitting on Stockfish's moves.
- **Decision:** Next I will implement training techniques to make the model more generalized.
- **AI help:** Claude Code wrote the `move_history` agent, the `--record` flag and
  `reencode.py`, generated and checked the data, trained both models, ran the evaluations,
  wrote the model card and drafted the facts in this entry.

### 2026-11-07: dropout, merging repeated positions, and branching games
*(Drafted by Claude Code from the run output; edited by Isaiah.)*
- **Question / hypothesis:** `move_history` seemed to be overfitting on Stockfish's moves, and
  most of the training data looked the same. Three changes should make the model generalize
  better: dropout; training on each repeated position once (Test A); and new games that
  branch into positions the old data never reached, later in the game (Test B).
- **Commit:** `a341617` (`--split view` and `--merge` in `baselines/imitation/train.py`) and
  `f307d6d` (`--random-plies`, `--keep-schedule`, `--matchups long` and `--play-agent` in
  `baselines/imitation/generate_data.py`). The default behaviour of both scripts is unchanged
  (checked byte for byte against the previous commit). The dropout models (`model.py` with
  `DROPOUT = 0.1`) are in `runs/`, which is not committed.
- **Setup:**
  - All models are `move_history` (29 planes, 64x4) trained for 16 epochs with
    `--save-best`, seed 0, about 12 min each on the RTX 3060 Ti.
  - Dropout: `nn.Dropout` after the first ReLU of each residual block and before the head
    (off when playing). Rates 0.1 and 0.3, on the `move_history` data.
  - Repeated positions, measured on the 75K-game data (baseline encoding): only 81.9% of
    positions are distinct. Repeats by stage: plies 0-3 100%, 4-9 62%, 10-19 2%, later
    about 1%; the start view alone appears 75,143 times. In 70.5% of repeated views the
    teacher gave different moves, so no model can match it on more than 58.6% of them.
  - Leak-free validation (`--split view`): the old split held out random positions, so
    common openings were in both training and validation. Holding out whole distinct views
    instead lowers the score of the same dropout-0.1 model from 42.5% to 41.3%. The models
    below are all compared with this split.
  - Test A (`--merge`): each distinct view trained once, with the share of each teacher move
    as a soft target. Under the 29-plane encoding only 1.0% of the 1,218,233 training views
    have more than one teacher move, so this mostly removes repeats.
  - Test B (branching games): 54,000 new games, seed 4, 1,508,144 positions (about 28 per
    game, against 20 before), 78 min on 10 workers. Each game opens with 0-10 random moves,
    positions are kept with probability 0.1 before ply 10, 0.5 before ply 20 and 1 after,
    and the matchups are guess-sf-4x500 vs itself, `move_history` vs guess-sf-4x500,
    `move_history` vs guess-sf-16x1000 and `move_history` vs itself. Opening positions
    (plies 0-9) fell from 25% to 4% of the data, and ply 20+ rose from 53% to 80%.
  - Evaluation: 200 games each against bot:greedy, bot:guess-sf-4x500 and
    bot:guess-sf-16x1000 on arena seed 6 (the same 600 games, with the same colours, as
    `move_history`), plus 1,000 games of Test B against the reference.
- **Result:**
  | Model | Validation match (own split) | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) |
  |---|---:|---:|---:|---:|---|
  | `move_history` (no dropout, earlier) | 42.0% (position split) | 173 | 173 | 135 | 1353 [1322, 1387] |
  | dropout 0.1, reference (`runs/mh_drop10_view`) | 41.3% | 170 | 166 | 138 | 1340 [1302, 1371] |
  | Test A, merged (`runs/mh_drop10_merge`) | 41.5% | 169 | 171 | 148 | 1367 [1330, 1402] |
  | Test B, branching (`runs/mh_drop10_branch`) | 39.1% (its own data) | 183 | 175 | 146 | **1401 [1359, 1437]** |

  Scores are out of 200.
  - **Head to head, Test B beat the reference 653 / 1000** (+110 Elo, 95% interval
    [+91, +141]). This is the first clearly significant gain since `baseline_75k`.
  - **Dropout alone did not help:** 42.5% (0.1) and 42.2% (0.3) against 42.0% on the old
    split, and the dropout-0.1 reference rated 13 below `move_history` in games (intervals
    overlap).
  - **Test A leans better** (+27 over the reference, intervals overlap; no head to head).
  - **Validation match did not predict the games.** Each model scored on both held-out sets:

    | Model | Old data, held out | Branching data, held out |
    |---|---:|---:|
    | reference | 41.3% | 37.8% |
    | Test A | 41.5% | 38.1% |
    | Test B | 40.4% | 39.1% |

    Test B matches Stockfish less often on old-style openings (48.8% vs 51.6%) and is level on
    later positions, yet it wins far more games. Matching about 38% of moves later in the game
    seems to be a ceiling while the teacher can see pieces the agent cannot.
  - **A likely reason:** most Test B games were played by `move_history` itself, so it
    trained on the positions it actually reaches in its own games, and on far more
    middlegame and endgame positions. The earlier data came only from bots playing each
    other. A different set of training games alone moved the rating by about 20 before, far
    less than this.
- **What surprised me:** I was surprised that dropout didn't help as I'd expect the model to generalize more. I believe that this may be because the model we're learning from is stockfish which sees the board when our model doesn't.
- **Decision:** Continue with branching learning.
- **AI help:** Claude Code added dropout, the view split, merging and the branching-game
  options, measured the repeated positions, generated the branching data, trained and
  evaluated the models and drafted the facts in this entry.

### 2026-11-07: fine-tuning on games against the bots (Idea 4)
*(Drafted by Claude Code from the run output; edited by Isaiah.)*
- **Question / hypothesis:** Training strictly to the teacher is not good, because Stockfish
  sees the whole board. Fine-tuning the best model on the results of its own Fog-of-War games
  against the reference bots should teach it what actually wins in the fog.
- **Commit:** `749e3b1` (`baselines/fow_games/play_games.py` and `encode_games.py`, the game
  dataset). `baselines/fow_games/finetune.py` is added in the same commit as this entry.
- **Setup:**
  - Dataset (`data/fow_games`, 52 min on 12 CPU workers, no Stockfish labels):
    `move_history_branch` played 60,000 games against bot:greedy, bot:guess-sf-4x500 and
    bot:guess-sf-16x1000 (20,000 each, half with each colour, dataset seed 100, so no game
    is shared with the arena seeds). Every agent turn is kept in order: 1,025,951 turns,
    median game 31 plies. It sampled every move at temperature 0.5 (600-game tests: 0.5
    stays close to its real strength while trying other moves; 1.0 played far worse). It
    scored 85% / 84% / 71% against the three bots, so 26% of turns come from lost games.
  - Fine-tuning (`finetune.py`): start from `move_history_branch`; each move gets the advantage
    A = (game result - the agent's average result against that opponent with that colour),
    optionally times gamma per turn before the end; plus 0.1 x KL(start || current) to stay
    close to the starting model; 2 epochs, AdamW lr 5e-5, batch 512, network in eval mode
    (no dropout, batch-norm frozen); 5% of games held out. About 70 s per run on the
    RTX 3060 Ti.
  - Methods: plain policy gradient; PPO-style clipped policy gradient (clip 0.2), with whole-game
    credit (gamma 1) and more credit near the end (gamma 0.95); advantage-weighted imitation
    (AWR, beta 0.5).
  - Evaluation: 200 games each against the three bots on arena seeds 6 and 7 (1,200 games),
    and 1,000 games head to head against `move_history_branch` on seeds 6 and 7.
- **Result:**
  | Model | Top move changed (held out) | Rating vs bots, 1,200 games | Head to head vs start, seed 6 / seed 7 | Both |
  |---|---:|---|---:|---:|
  | `move_history_branch` (start) | - | 1414 [1388, 1443] | - | - |
  | plain policy gradient | diverged | - | - | - |
  | PPO, gamma 1 | 5.5% | 1421 [1393, 1453] | 524 / 476.5 | 50.0% |
  | PPO, gamma 0.95 | 5.2% | 1417 [1385, 1446] | 531.5 / 492 | 51.2% |
  | AWR | 5.8% | 1420 [1390, 1448] | 489 / - | 48.9% |

  Head-to-head scores are the fine-tuned model's, out of 1,000.
  - **Plain policy gradient diverged:** it pushed the moves of lost games down without limit
    (held-out KL 5.0 after one epoch, then values in the millions). Clipping fixed it.
  - **No gain in games:** every fine-tuned model is within a few points of the start against
    the bots, and head to head over 2,000 games they are even (the best, +8 Elo, is within
    noise). The PPO gamma-0.95 model looked better after the first 1,000 games (531.5) but
    not after the second (492).
  - **It did learn something small that carries over:** on held-out games, PPO with gamma 1
    made moves from won games more likely (+0.035 log-probability) and moves from lost games
    less likely (-0.016). AWR raised both by about +0.2: it mostly copied its own sampled moves.
  - **Likely reasons:** one result per game is spread over about 17 moves, and the baseline
    only knows the opponent and colour, not how good the position was, so most of the signal
    is noise; the clipping keeps changes small (the top move changed in about 5% of
    positions); only about 15,000 lost games.
- **What surprised me:** I was surprised the dataset didn't help train the model better.
- **Decision:** Stop this version and add a value head: a network that predicts the result of
  the game from each position, so each move is judged against how good its position was.
- **AI help:** Claude Code wrote the dataset generator and `finetune.py`, generated the games,
  ran the fine-tuning and evaluations and drafted the facts in this entry.

### 2026-11-07: fine-tuning with a value head
*(Drafted by Claude Code from the run output; edited by Isaiah.)*
- **Question / hypothesis:** Fine-tuning on game results did not help because one result per
  game is spread over all its moves. A value head that predicts the result from each position
  should let each move be judged against how good its position was.
- **Commit:** data at `749e3b1`; `baselines/fow_games/value.py` and the `--value` / `--lam`
  options of `finetune.py` are added in the same commit as this entry.
- **Setup:**
  - Value head (`value.py`): on the frozen trunk of `move_history_branch` (stem and residual
    blocks), conv 1x1 to 32 channels, 2048 -> 128, plus a learned vector for the opponent
    (known during training), -> 1 with tanh; 264,897 trainable parameters; trained 3 epochs to
    predict the result (MSE), on the same games and held-out split as the fine-tuning. Used
    only during training, so the agent does not change size.
  - Fine-tuning: as in the previous entry (PPO clip 0.2, KL 0.1, 2 epochs, lr 5e-5), but each
    move's advantage is generalized advantage estimation from the value head:
    delta = V(next own turn) - V(this turn), with the game result after the last turn, and
    A = sum of (lambda)^k delta over the following turns. lambda 1 = result minus the
    position's value; lambda 0.8 leans on the change in value over the next few turns.
  - Evaluation: 200 games each against the three bots on arena seeds 6 and 7, and 1,000
    games head to head against `move_history_branch` per seed.
- **Result:**
  - **The value head is weak but helps:** held-out error 0.638 against 0.730 for the
    opponent-and-colour baseline; it calls the result right 77.9% of the time (always guessing
    a win: 74%). Without the opponent input: 0.648; training a copy of the trunk too: 0.645.
    Its error is lowest early in the game (0.59 before ply 20) and highest late (0.71).

  | Model | Rating vs bots, 1,200 games | Head to head vs `move_history_branch` (per 1,000) | Total |
  |---|---|---|---:|
  | `move_history_branch` (start) | 1414 [1388, 1443] | - | - |
  | value head, lambda 1 | 1437 [1408, 1467] | 524 / 512 (seeds 6, 7) | 51.8% of 2,000 |
  | **value head, lambda 0.8** (`runs/ft_vppo_l08`) | 1428 [1402, 1457] | **550 / 531.5 / 552** (seeds 6, 7, 8) | **54.5% of 3,000** |

  - **lambda 0.8 is a clear gain:** 1,633.5 / 3,000 head to head, +31 Elo (95% interval
    about +18 to +44), consistent on all three seeds. Against the bots both value-head models
    lean higher than the start, but the intervals overlap.
  - **lambda 1 is within noise** (+12 Elo over 2,000 games), like the fine-tuning without a
    value head. Judging a move by the change in value right after it, rather than by the
    final result, is what made the difference.
  - The changes are still small: the top move changed in 7.2% of held-out positions.
- **What surprised me:** Position and context are better things to train on then final game result.
- **Decision:** I'm not a massive fan of value head but it improves elo so i'll explore it more.
- **AI help:** Claude Code wrote `value.py` and the value-head option of `finetune.py`, trained
  and evaluated the models and drafted the facts in this entry.

### 2026-11-15: a transformer over the game's turns (Idea 3), and how much data helps
*(Facts drafted by Claude Code from the run output; surprises and decision to be written by Isaiah.)*
- **Question / hypothesis:** The CNN sees one turn at a time; its only memory is the hand-made
  planes. A small transformer that reads a summary of every earlier turn of the game should let
  the agent use what it saw before: pieces that appeared and vanished into the fog, where the
  opponent has been active.
- **Commit:** `agents/turn_transformer/` and `baselines/sequence/` are added in the same commit as
  this entry; the data is `move_history_branch`'s branching games (see its model card).
- **Setup:**
  - Model (`agents/turn_transformer/model.py`, 805,802 parameters, under the 1M size floor):
    `move_history_branch`'s CNN reads each turn; each turn is squeezed into 128 numbers; a
    2-layer causal transformer (4 heads) reads those for the agent's last 64 turns at most; its
    output scales and shifts the current turn's CNN features before the move head (starts at
    zero, so training starts from a plain CNN). 4.4e7 FLOPs per move on average, VALID.
  - Data (`encode_seq.py`): the same 54,000 branching games, replayed in order, one sequence per
    side: 108,000 sequences, 2.25M turns, 1.51M with a Stockfish label (the rest are context).
  - Training (`train_seq.py`): 16 epochs, loss on the labelled turns, 5% of games held out; a
    control trained the same CNN without the transformer on exactly the same batches.
  - Evaluation: 200 games against each bot on arena seeds 6 and 7; 1,000 games head to head per
    seed.
- **Result:**

  | Model | Held-out match | Rating vs bots, 1,200 games |
  |---|---:|---|
  | control (no transformer) | 39.3% | 1377 [1348, 1403] |
  | `move_history_branch` (same games, earlier recipe) | - | 1414 [1385, 1440] |
  | value-head fine-tune (`runs/ft_vppo_l08`) | - | 1428 [1401, 1458] |
  | **turn_transformer** | **40.1%** | **1450 [1422, 1482]** |

  - Head to head, the transformer scored 1087 / 2000 against the control (54.4%, about +30
    Elo), 1194.5 / 2000 against `move_history_branch` (59.7%, about +68) and 1138 / 2000 against
    the value-head fine-tune (56.9%, about +48). Both seeds agreed each time.
  - The control rated below `move_history_branch` with the same network and games: batches
    built from whole games and holding out whole games seem to train a slightly worse plain
    CNN. The comparison with the transformer is still fair, because both were trained that way.
- **How much data helps** (same held-out games and number of steps):

  | Training games | Held-out match | Best epoch | Rating, 1,200 games |
  |---|---:|---|---|
  | 25% (12,825) | 37.7% | 9 of 64 | 1369 [1347, 1393] |
  | 50% (25,650) | 39.0% | 10 of 32 | 1393 [1368, 1422] |
  | 100% (51,300) | 40.1% | 11 of 16 | 1450 [1422, 1482] |
  | 100% + 75,000 older games | 40.9% | 15 of 16 | 1461 [1430, 1493] |

  - More of the agent's own branching games still helps, and is not flattening yet. The
    smaller runs overfit hard: the 25% run fell from 37.7% to 33.5% after its peak.
  - The 75,000 older games (`data/rec75k_s3`, played by `move_history` without branching) raised
    the teacher match but lost head to head against the model without them, 927 / 2000. Data
    from positions the agent actually reaches matters more than more teacher answers.
- **What surprised me:** I was not that surprised that a transformer increased performance.
- **Decision:** Grow the transformer and learn from FoW games instead of a stock fish teacher.
- **AI help:** Claude Code wrote the sequence encoder, the turn-transformer model and agent, the
  training script and its data options, trained and evaluated the models, wrote the model card
  and drafted the facts in this entry.

### 2026-11-16: deeper CNN, and learning from FoW games instead of the teacher
*(Facts drafted by Claude Code from the run output; surprises and decision to be written by Isaiah.)*
- **Question / hypothesis:** (1) Now that the model reads whole games, does a deeper CNN help?
  (2) Imitating Stockfish is capped by a teacher who sees the whole board. Can the turn
  transformer improve by learning only from the results of its own Fog-of-War games, without
  overfitting to old games or to the three bots?
- **Commit:** `agents/turn_transformer_rl/`, `baselines/fow_games/rl_loop.py` and
  `baselines/guard.py` are added with this entry.
- **Deeper CNN:** 6 residual blocks instead of 4 (954,286 parameters), same data and settings.
  Held-out match 40.4% (4 blocks: 40.1%), rating 1431 [1404, 1464] (4 blocks: 1450), head to
  head 1019 / 2000 against `turn_transformer`: a tie. Kept 4 blocks.
- **Setup of the learning loop** (`rl_loop.py`, details in `baselines/fow_games/README.md` and
  the model card):
  - Start and anchor: `agents/turn_transformer`. No Stockfish labels.
  - Each round: 6,000 new games, 75% against the three bots and 25% against the last 3 saved
    rounds; the learner samples at temperature 0.5. Buffer: the last 2 rounds of games.
  - PPO (clip 0.2) with advantages from a value network (GAE λ 0.8), plus 0.1 × KL(start ‖
    current). The value network was warmed up on 60,000 recorded games (`data/fow_games`).
  - Every 2 rounds: 600 games against the bots on seed 9 to pick the best round; stop after 3
    ratings without a new best. Then confirm on seeds 6 and 7 and head to head.
  - The PC blue-screened on the first attempt (game workers and GPU training at the same time).
    Since then every long job holds a lock and pauses while memory is low (`baselines/guard.py`),
    and the loop resumes from its last finished round (`--resume`).
- **Result:**

  | Round | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Rating, seed 9 (600 games) | 1434 | 1498 | 1542 | **1550** | 1522 | 1512 | 1522 |

  - Round 8 (`agents/turn_transformer_rl`) on seeds 6 and 7: **1519 [1488, 1553]** (379 / 374 /
    339 of 400) against `turn_transformer`'s 1450 [1422, 1482]. Head to head against
    `turn_transformer`: 619 + 633 = **1252 / 2000 (62.6%, about +90 Elo)**.
  - The KL to the starting model reached only 0.04; 3-5% of ratios were clipped; no draws.
- **Overfitting to the bots?** The gains were larger against agents it never played while
  learning (1,000 games each on seed 6): 736 against `move_history_branch`, 694 against the
  value-head fine-tune, 701 against the transformer's control, 905.5 against `agents/baseline`.
  Against bots it never played: 199 / 200 against bot:random, 184 against bot:naive-sf, 178
  against bot:guess-sf-32x4000 (more search did not make that bot stronger).
- **Where the losses come from** (every lost game against the bots, 1,200 games per model,
  classified by the position before our last move):

  | Model | Losses | Already attacked by a hidden piece | King stepped into a hidden attack | Left attacked by a visible piece | Another move opened a line |
  |---|---:|---:|---:|---:|---:|
  | turn_transformer_rl | 108 | 64 | 29 | 9 | 6 |
  | turn_transformer | 152 | 58 | 80 | 3 | 11 |
  | value-head fine-tune | 169 | 75 | 71 | 12 | 11 |
  | move_history_branch | 181 | 81 | 80 | 4 | 16 |
  | control | 214 | 95 | 96 | 9 | 14 |

  No model ever missed a chance to capture the enemy king or stepped into an attack it could
  see. About half of every model's losses are against guess-sf-16x1000, and about 85% come
  after ply 20. Learning from its own games mostly removed the king-steps-into-a-hidden-attack
  losses (80 to 29).
- **What surprised me:** Training on FoW games increased performance greatly.
- **Decision:** Loss mostly comes from late games- so I will fine-tune for that.
- **AI help:** Claude Code wrote the learning loop, the guard and the loss analysis, ran the
  experiments one at a time, wrote the model card and drafted the facts in this entry.

### 2026-11-17: fine-tuning on the positions it loses
*(Drafted by Claude Code from the run output; reviewed by Isaiah.)*
- **Question / hypothesis:** `turn_transformer_rl` loses mostly late in the game (1.94 losses per
  100 plies after ply 50, against 0.08 in plies 1-20), and games that start from move 1 rarely
  reach those positions. If half the learning games start just before positions where it lost,
  it should lose less late, and gain rating.
- **Commit:** the commit after `b75ecf1`: `rl_loop.py` gains `--midgame-lost` (start 10-30 plies
  before a recorded loss, playing the losing side against the bot that won) and `--bot-weights`.
- **Setup:**
  - Start: `agents/turn_transformer_rl`, with the value network from its loop. The KL penalty
    now pulls towards `turn_transformer_rl`.
  - Each round: 6,000 games, half replaying one of 13,376 lost games of at least 30 plies from the
    first loop (`runs/rl_tf`, 84,000 games), half from move 1. Bot games: 50% guess-sf-16x1000,
    25% each greedy and guess-sf-4x500; 25% of all games against its last 3 rounds.
  - Same rating and stopping rule as before: 600 games on seed 9 every 2 rounds, stop after 3
    ratings without a new best. Then seeds 6 and 7, and 2,000 games head to head.
  - Compute: 97 minutes for 10 rounds on the RTX 3060 Ti PC (7-8 CPU workers for games, under a
    minute of GPU training per round), plus 5 minutes of checks. Not Sol.
- **Result:**

  | Round | 2 | 4 | 6 | 8 | 10 |
  |---|---:|---:|---:|---:|---:|
  | Rating, seed 9 (600 games) | 1542 | **1597** | 1550 | 1572 | 1542 |
  | Against `move_history_branch` (never trained against) | 75.5% | 76% | 76% | 74.5% | 75.5% |

  - Round 4 (`agents/turn_transformer_late`) on seeds 6 and 7: **1580 [1544, 1625]** (376 / 380 /
    365.5 of 400) against `turn_transformer_rl`'s 1519 [1488, 1553] (379 / 374 / 339). Head to
    head against `turn_transformer_rl`: 616 + 599 = **1215 / 2000 (60.75%, about +75 Elo)**.
  - Losses in the same 1,200 bot games: 108 to 78. Per 100 plies: plies 1-20 0.08 to 0.05,
    plies 21-50 0.38 to 0.34, after ply 50 1.94 to 1.35. Against guess-sf-16x1000: 61 to 34.
  - By cause: king already attacked by a hidden piece 64 to 52 (now 67% of losses), king stepped
    into a hidden attack 29 to 18, left attacked by a visible piece 9 to 4, a move opened a line 6
    to 4.
  - The KL to the starting model reached 0.03 at round 4; 4-9% of ratios clipped.
  - The guard paused 4 times when free memory fell below 1 GB (about 15 s each); the next runs
    on this PC should use 10 workers instead of 12.
- **What surprised me:** I was surprised at the amount of elo that it gained, but wonder why it plateaus. 2/3 of losses are still from late end-games.
- **Decision:** It replaced `turn_transformer_rl` as the entry (`submission/`), and
  `reproduce.sh` now has it as stage 7. I plan to add a 'danger' heatmap so the king doesn't get put into check as much from possible dangerous squares.
- **AI help:** Claude Code added the two options to the learning loop, ran the fine-tune and the
  checks one job at a time, wrote the model cards and drafted the facts in this entry.
