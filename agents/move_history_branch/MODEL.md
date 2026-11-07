# Model card: move_history_branch

- **Name:** `move_history` trained on branching games, with dropout (residual CNN, no search), random openings
- **Core model:** `model.py` `PolicyNet(in_planes=29, channels=64, blocks=4)`; scores every (from, to) square pair. Same network as `agents/move_history` plus dropout 0.1 after the first convolution of each residual block and before the head (`DROPOUT` in `model.py`; training only, `agent.py` calls `.eval()`). A fixed, unsaved buffer scales the turn-number plane (22) by 1/100
- **Parameters:** 318,089 as counted by `python -m fow.params agents/move_history_branch` (316,928 trainable weights plus batch-norm statistics stored in the file)
- **Weight files and SHA-256:** `weights/model.safetensors`, `b7fd0dca8be9c4f3f80862ddd71daf24d9688867841b37e86b81806eac6ea3d7`
- **Input:** 29 planes, exactly as `agents/move_history` (`features.py`, `history.py`): the baseline's 22 planes (`fow.memory.Memory` for remembered pieces), plane 22 = ply number capped at 100, planes 23-28 = from and to squares of this agent's own last 3 moves, flipped for Black
- **Play:** the first 4 own moves are sampled from `softmax(logits / 1.0)` over legal moves using the game seed; after that, the top-scoring legal move. See `agent.py`
- **Training data:** 1,508,144 positions from 54,000 branching games (seed 4), labelled by Stockfish 19 at 5,000 nodes on the full board. Each game opens with 0-10 random moves; positions are kept with probability 0.1 before ply 10, 0.5 before ply 20 and 1 after; the matchups are guess-sf-4x500 vs itself, `move_history` vs guess-sf-4x500, `move_history` vs guess-sf-16x1000 and `move_history` vs itself, where `move_history` is `agents/move_history` with its committed weights (SHA-256 `671d979f...f73492`). 4% of positions are from plies 0-9 and 80% from ply 20 on. Command:
  `python baselines/imitation/generate_data.py --games 54000 --workers 10 --seed 4 --out data/branch54k_s4 --agent-dir agents/move_history --record --random-plies 10 --keep-schedule "0:0.1,10:0.5,20:1" --matchups long --play-agent agents/move_history`; the shards were then moved to `data/branch54k_s4_hist`
- **Training:** `python baselines/imitation/train.py --data data/branch54k_s4_hist --agent-dir agents/move_history_branch --channels 64 --blocks 4 --epochs 16 --seed 0 --device cuda --save-best --split view` (AdamW, one-cycle learning rate 2e-3, batch 256, weight decay 1e-4); the weights are from epoch 15, the best validation epoch. Code at commits `a341617` and `f307d6d`
- **Training compute:** 78 minutes of data generation on 10 CPU workers and 12 minutes of training on one RTX 3060 Ti (about 0.2 GPU-hours), on a personal Windows PC, not Sol
- **Validation match with teacher:** 39.1% of held-out views of its own data (whole distinct views held out, so none also appear in training); 40.4% on held-out views of `move_history`'s data, where `move_history`'s recipe with the same dropout scores 41.3%
- **Validity:** `python -m fow.validate agents/move_history_branch --device cuda` prints VALID: 4.04e7 FLOPs per move, same moves on replay, 10/10 wins against bot:random
- **Local rating:** 200 games each against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000. The guess-sf bots do not play the same moves on every run, so the same seed gives slightly different results each time:

  | Run | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) |
  |---|---:|---:|---:|---|
  | seed 6 (as `runs/mh_drop10_branch`) | 183 | 175 | 146 | 1401 [1359, 1437] |
  | seed 6 (this directory) | 183 | 171 | 155 | 1412 [1378, 1451] |
  | seed 7 | 176 | 175 | 159 | 1415 [1374, 1457] |
  | all 1,800 games | 542 / 600 | 521 / 600 | 460 / 600 | 1409 [1387, 1431] |

  On seed 6, `move_history` rated 1353 and the same recipe with dropout on `move_history`'s data 1340. Head to head against that dropout model: 653 / 1000 (+110 Elo, 95% interval [+91, +141]).
- **How to reproduce:** not yet in `reproduce.sh`. It needs `agents/move_history` (committed), the data command above, then the training command above with `model.py` from this directory
