# Model card: move_history

- **Name:** imitation CNN told the turn number and its own last 3 moves (residual CNN, no search), random openings
- **Core model:** `model.py` `PolicyNet(in_planes=29, channels=64, blocks=4)`; scores every (from, to) square pair. Same architecture as `agents/baseline_75k` except that the first layer takes 29 planes instead of 22; a fixed, unsaved buffer scales the turn-number plane (22) by 1/100
- **Parameters:** 318,089 as counted by `python -m fow.params agents/move_history` (`baseline_75k`'s network: 314,057)
- **Weight files and SHA-256:** `weights/model.safetensors`, `671d979f3cf1cdbe5d834708498562a09f986fb244381d538649198720f73492`
- **Input:** 29 planes; see `features.py`. Planes 0-21 are exactly the baseline's (`fow.memory.Memory` for remembered pieces). Plane 22 holds the ply number, capped at 100. Planes 23-28 mark the from and to squares of this agent's own last 3 moves, most recent first (`history.py`), flipped for Black like the rest of the board; they are empty until those moves exist
- **Play:** the first 4 own moves are sampled from `softmax(logits / 1.0)` over legal moves using the game seed; after that, the top-scoring legal move. See `agent.py`
- **Training data:** 1,514,142 positions from 75,000 new bot-vs-bot games (seed 3), labelled by Stockfish 19 at 5,000 nodes on the full board. Generated with `baselines/imitation/generate_data.py --games 75000 --workers 12 --seed 3 --out data/rec75k_s3 --agent-dir agents/move_history --record`; the shards were moved to `data/rec75k_s3_hist`. The recorded games (`data/rec75k_s3/games_*.jsonl.gz`) were re-encoded with `baselines/imitation/reencode.py` into the baseline encoding as a control (`data/rec75k_s3_base`, identical planes 0-21, labels and legal masks)
- **Training:** `baselines/imitation/train.py --data data/rec75k_s3_hist --agent-dir agents/move_history --channels 64 --blocks 4 --epochs 16 --seed 0 --device cuda --save-best` (AdamW, one-cycle learning rate 2e-3, batch 256); the weights are from epoch 9, the best validation epoch
- **Training compute:** about 50 minutes of data generation on 12 CPU workers, 11 minutes of training on one RTX 3060 Ti
- **Validation match with teacher:** 42.0% of held-out positions (epoch 9). The control (baseline encoding, same data and settings) also reached 42.0% at epoch 9
- **Local rating:** 200 games each against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000, arena seed 6:

  | Model | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) |
  |---|---:|---:|---:|---|
  | `move_history` | 173 | 173 | 135 | 1353 [1322, 1387] |
  | control (baseline encoding, same data) | 169 | 165 | 135 | 1331 [1299, 1365] |
  | `baseline_75k` | 165 | 170 | 143 | 1348 [1310, 1374] |

  Head to head against the control: 526.5 / 1000 (+18 Elo, 95% interval [-5, +35]; not significant).
