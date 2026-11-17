# Model card: submission (turn_transformer_late)

- **Tournament device:** cpu
- **Entry:** a copy of `agents/turn_transformer_late` (same code, config and weights). One CPU thread (`torch.set_num_threads(1)` in `agent.py`); it also runs on cuda and plays the same moves there (8 of 8 test games identical)
- **Name:** `agents/turn_transformer_rl` after 4 more rounds of learning from its own Fog-of-War games, half of them started from positions where an earlier version went on to lose (PPO with a value network, no teacher moves, no search)
- **Core model:** identical to `agents/turn_transformer` and `agents/turn_transformer_rl` (same `model.py`, `agent.py`, `features.py`, `history.py` and `config.json`): `PolicyNet(in_planes=29, channels=64, blocks=4, d_model=128, layers=2, heads=4, context=64)`, a residual CNN per turn plus a 2-layer causal transformer over the agent's last 64 turns. Only the weights differ. See `agents/turn_transformer` for the architecture
- **Parameters:** 805,802 as counted by `python -m fow.params submission`; under the 1M size floor
- **Weight files and SHA-256:** `weights/model.safetensors`, `78ee2942044035eb03346c481eace8ea5cbdc231adf515a8ebdff117e52c28ca` (round 4 of `runs/rl_late`)
- **Input, state within a game, play:** as `agents/turn_transformer`: 29 planes per turn, the turn vectors of this game only (cleared in `reset`), the first 4 own moves sampled at temperature 1 with the game seed, then the top-scoring legal move
- **Starting point and anchor:** `agents/turn_transformer_rl` (the KL penalty pulls towards it, not towards `turn_transformer`)
- **Why:** 87% of `turn_transformer_rl`'s losses came after ply 20, and its losses per 100 plies rose from 0.08 in plies 1-20 to 1.94 after ply 50. Games that start from move 1 rarely reach those positions, so the loop rarely learned from them
- **Training data:** only games the model played itself, 6,000 new games per round (`baselines/fow_games/rl_loop.py --midgame-lost`):
  - Half the games (about 3,000 per round) replay a recorded game that an earlier model lost by having its king captured: the first moves are played as recorded, up to a point 10 to 30 plies before the loss (and not before ply 20), then the model plays on from there as the losing side, against the bot it lost to. The source games are the 84,000 games of `runs/rl_tf` (the loop that produced `turn_transformer_rl`); 13,376 of them are such losses of at least 30 plies. The replayed moves give the model its memory of the game but are not learned from
  - The other half start from move 1
  - Opponents: 75% bots, of which 50% bot:guess-sf-16x1000 (the bot it loses to most) and 25% each bot:greedy and bot:guess-sf-4x500; 25% its last 3 saved rounds. Each opponent gets both colours equally (a replayed game keeps the losing side's colour)
  - Rounds 1 to 4: 24,000 games, 545,000 turns of which 337,000 were learned from (the rest were replayed moves). Learner moves sampled at temperature 0.5. Buffer: the last 2 rounds. No Stockfish labels. Game seeds never overlap the evaluation seeds (6, 7, 9)
- **Value network (training only, not part of the agent):** the one `runs/rl_tf` ended with (`--value-init runs/rl_tf/value.safetensors`), trained one pass per round on the buffer. Its error on each round's new games fell from 0.88 in round 1 (positions it had not seen before) to 0.77 in round 4
- **Training:** as `turn_transformer_rl`: per round, 2 passes of PPO over the buffer, GAE (lambda 0.8, gamma 1), ratio clip 0.2, a penalty of 0.1 x KL(`turn_transformer_rl` || current), learning rate 5e-5, each game's turns weighted min(1, 40 / its turns). KL to the starting model after round 4: 0.033; 4-9% of ratios clipped per round
- **Choosing the round:** each even round was rated against the three bots on seed 9 (600 games): rounds 2, 4, 6, 8, 10 rated 1542, **1597**, 1550, 1572, 1542, so the loop stopped after round 10. On the same seed `turn_transformer_rl` had rated 1550. Because the round was picked on seed 9, the ratings below use seeds 6 and 7 only
- **Training compute:** 37 minutes for rounds 1 to 4 (97 minutes for all 10 rounds including the ratings), game play on 7 to 8 CPU workers plus under a minute of GPU training per round, on one RTX 3060 Ti PC, not Sol
- **Validity:** `python -m fow.validate submission --device cpu --games 10` prints VALID: mean 4.41e7, max 4.94e7 FLOPs per move, 11.2 ms per move on average on one CPU thread of this PC, loaded in 1.8 s. The cost per move does not depend on the weights; the transformer reads at most 64 turns, so it is bounded (the largest seen in 1,800 recorded games was 9.3e7)
- **Local rating:** 200 games each against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000:

  | Run | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) |
  |---|---:|---:|---:|---|
  | seed 6 | 191 | 187 | 184.5 | 1589 [1549, 1650] |
  | seed 7 | 185 | 193 | 181 | 1572 [1514, 1628] |
  | all 1,200 games | 376 / 400 | 380 / 400 | 365.5 / 400 | 1580 [1544, 1625] |

  On the same seeds, `agents/turn_transformer_rl` rated 1519 [1488, 1553] (379 / 374 / 339)
- **Head to head** against `agents/turn_transformer_rl` (1,000 games per seed; colours alternate): 616 on seed 6, 599 on seed 7, **1215 / 2000 (60.75%, about +75 Elo)**. Against `agents/move_history_branch`, which it never played while learning: 75.5-76% at every rating (200 games each, seed 9), about the same as `turn_transformer_rl`'s 73.6%
- **Where it still loses** (78 losses in the 1,200 bot games, against 108 for `turn_transformer_rl`):

  | | `turn_transformer_rl` | `turn_transformer_late` |
  |---|---:|---:|
  | Losses per 100 plies, plies 1-20 | 0.08 | 0.05 |
  | plies 21-50 | 0.38 | 0.34 |
  | plies 51 and later | 1.94 | 1.35 |
  | Losses against guess-sf-16x1000 | 61 | 34 |
  | King already attacked by a piece it could not see | 64 | 52 |
  | King stepped onto a square attacked by a piece it could not see | 29 | 18 |
  | King left attacked by a piece it could see | 9 | 4 |
  | Another move opened a line to the king | 6 | 4 |

  The late game is still where most losses happen, and checks from pieces it cannot see are now two thirds of them
- **Hardware and GPU-hours:** everything ran on a personal Windows PC (RTX 3060 Ti, 16 CPU threads), not Sol. GPU: about 1.7 GPU-hours in total for the whole chain (training `move_history`, `move_history_branch` and `turn_transformer`, both learning loops' training steps and the rating games); the rest is CPU game play and Stockfish labelling, about 6.5 hours on 7-12 CPU workers
- **How to reproduce:** `bash reproduce.sh` from a fresh clone (see `REPRODUCE.md`). It rebuilds every stage: data for `move_history`, its training, the branching games, `move_history_branch`, `turn_transformer`, the value warm-up games, the first learning loop (`runs/rl_tf`, which gives `turn_transformer_rl`), this loop from that one's lost games (`runs/rl_late`), then copies the best round into `submission/`. The bots are not deterministic, so the rebuilt weights will differ from these
