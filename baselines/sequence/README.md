# Sequence training (PLAN.md, Idea 3)

Training for `agents/turn_transformer`: `move_history_branch`'s CNN plus a small causal
transformer that reads the game turn by turn, so each move can depend on what the agent saw on
earlier turns (pieces that appeared and vanished into the fog, where the opponent has been
active), not only on the current board and the hand-made memory planes.

| Script | What it does |
|---|---|
| `encode_seq.py` | replays games recorded with `baselines/imitation/generate_data.py --record` and encodes every turn of each side in order (two sequences per game); turns with a Stockfish label keep it |
| `train_seq.py` | trains the policy on those sequences, with the loss on the labelled turns; `--no-context` trains the same CNN without the transformer on the same batches, as a control |

```bash
python baselines/sequence/encode_seq.py --games data/branch54k_s4 --out data/branch54k_s4_seq      # about 1 min
python baselines/sequence/train_seq.py --data data/branch54k_s4_seq --agent-dir agents/turn_transformer
python baselines/sequence/train_seq.py --data data/branch54k_s4_seq --agent-dir runs/seq_control --no-context
```

Two options for data experiments; both keep exactly the same held-out games, so their match
numbers stay comparable:
- `--games-frac 0.5 --epochs 32` trains on half of the training games, with the same number of
  steps.
- `--extra-data data/rec75k_s3_seq` adds more sequence directories, for training only.

Results with the turn transformer:

| Training games | Held-out match | Rating |
|---|---|---|
| 25% | 37.7% | 1369 |
| 50% | 39.0% | 1393 |
| All | 40.1% | 1450 |
| All, plus `rec75k_s3`'s 75,000 older games | 40.9% | 1461, but 927 / 2000 head to head against the model trained on all |

The older games were played by `move_history` without branching.

## Model (`agents/turn_transformer/model.py`)

- **Per turn:** the 64x4 CNN of `move_history_branch` reads the 29 input planes (what is in view
  now, including visible enemy pieces, remembered enemy pieces, the turn number and the agent's
  last 3 moves).
- **Turn summary:** each turn's CNN features are squeezed into one vector of 128 numbers.
- **Transformer:** 2 causal layers, width 128, 4 heads, over at most the last 64 own turns. Each
  turn sees itself and the turns before it.
- **Back into the move choice:** the transformer's output scales and shifts the current turn's
  CNN features before the move head. This starts at zero, so training starts from a plain CNN.
- **Size:** 805,802 parameters (`fow.params`), under the 1M size-bonus floor; about 5e7 FLOPs per
  move. The agent keeps the game's turn vectors and clears them in `reset`.

Checked: replaying a recorded game through the agent turn by turn gives the same move scores as
the batch computation used in training (to within 1e-5), and `fow.validate` reports VALID.
