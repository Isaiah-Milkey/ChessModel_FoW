# Model card: submission (turn_transformer_rl)

- **Tournament device:** cpu
- **Entry:** a copy of `agents/turn_transformer_rl` (same code, config and weights). One CPU thread (`torch.set_num_threads(1)` in `agent.py`); it also runs on cuda and plays the same moves there (8 of 8 test games identical)

- **Name:** `agents/turn_transformer` after 8 rounds of learning from its own Fog-of-War games (PPO with a value network, no teacher moves, no search)
- **Core model:** identical to `agents/turn_transformer` (same `model.py`, `agent.py`, `features.py`, `history.py` and `config.json`): `PolicyNet(in_planes=29, channels=64, blocks=4, d_model=128, layers=2, heads=4, context=64)`, the residual CNN of `agents/move_history_branch` per turn plus a 2-layer causal transformer over the agent's last 64 turns. Only the weights differ. See that card for the architecture
- **Parameters:** 805,802 as counted by `python -m fow.params submission`; under the 1M size floor
- **Weight files and SHA-256:** `weights/model.safetensors`, `dbae26c11bf4e1eb2409a0d2dfab9d9541d94c090db807b5636c525a83ee1efb` (round 8 of `runs/rl_tf`)
- **Input, state within a game, play:** as `agents/turn_transformer`: 29 planes per turn, the turn vectors of this game only (cleared in `reset`), the first 4 own moves sampled at temperature 1 with the game seed, then the top-scoring legal move
- **Starting point and anchor:** `agents/turn_transformer` (trained to imitate Stockfish 19 on 54,000 branching games; see its card)
- **Training data:** only games the model played itself, 6,000 new games per round (`baselines/fow_games/rl_loop.py`): 75% against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000, 25% against its last 3 saved rounds (the starting model included at first), each opponent with both colours equally. The learner samples its moves at temperature 0.5. A round learns from its own games and the previous round's (12,000 games, about 200,000 of its turns); older games are dropped, so each game is used in 2 rounds (4 passes) and never again. Rounds 1 to 8: 48,000 games, 810,000 turns. No Stockfish labels. The game seeds never overlap the evaluation seeds (6, 7, 9)
- **Value network (training only, not part of the agent):** a copy of the policy's CNN and transformer with its own head, which also sees which opponent it faces; it predicts the final result (+1/-1) at every turn. Warmed up on 60,000 recorded games of `move_history_branch` against the bots (`data/fow_games`, 1.03M turns, train MSE 0.635), then trained one pass per round on the buffer (MSE on each round's new games fell from 0.69 in round 1 to 0.58 in round 7)
- **Training:** per round, 2 passes of PPO over the buffer: advantages from GAE (lambda 0.8, gamma 1) on the value network, ratio clipped at 0.2 against the network that played the game, a penalty of 0.1 x KL(starting model || current), learning rate 5e-5, each game's turns weighted min(1, 40 / its turns). Played and trained move probabilities agree to within 1e-5 (TF32 is turned off on the GPU). KL to the starting model after round 8: 0.039; 3-5% of ratios clipped per round
- **Choosing the round:** each even round was rated against the three bots on seed 9 (600 games). Round 8 rated highest (1550 [1508, 1603]); rounds 10, 12 and 14 rated 1522, 1512 and 1522, so the loop stopped there (3 ratings without a new best). Because the round was picked on seed 9, the ratings below use seeds 6 and 7 only
- **Training compute:** about 1.5 hours for rounds 1 to 8 (game play on 6 to 8 CPU workers plus under a minute of GPU training per round) on one RTX 3060 Ti PC, not Sol; plus the value warm-up (2 minutes) and the training of `agents/turn_transformer`
- **Validity:** `python -m fow.validate submission --device cpu --games 10` prints VALID: mean 4.62e7, max 5.93e7 FLOPs per move, 11.8 ms per move on average on one CPU thread of this PC, loaded in 10.9 s. Across 1,800 recorded games against the bots (up to 124 plies) the largest move cost 9.3e7 FLOPs; the transformer reads at most 64 turns, so the cost per move is bounded
- **Local rating:** 200 games each against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000 (the guess-sf bots do not play the same moves on every run):

  | Run | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) |
  |---|---:|---:|---:|---|
  | seed 6 | 192 | 188 | 167 | 1522 [1485, 1573] |
  | seed 7 | 187 | 186 | 172 | 1515 [1458, 1570] |
  | all 1,200 games | 379 / 400 | 374 / 400 | 339 / 400 | 1519 [1488, 1553] |

  On the same seeds, `agents/turn_transformer` rated 1450 [1422, 1482] (364 / 371 / 313). Against other bots on seed 6 (200 games each): bot:random 199, bot:naive-sf 184, bot:guess-sf-32x4000 178
- **Head to head** (1,000 games per seed; colours alternate):

  | Opponent | Seed 6 | Seed 7 | Score |
  |---|---:|---:|---:|
  | `agents/turn_transformer` (its starting point) | 619 | 633 | 1252 / 2000 (62.6%) |
  | `runs/ft_vppo_l08` (`move_history_branch` after one round with a value head) | 694 | | 694 / 1000 |
  | control for turn_transformer (same CNN, no transformer) | 701 | | 701 / 1000 |
  | `agents/move_history_branch` | 736 | | 736 / 1000 |
  | `agents/move_history` | 807.5 | | 807.5 / 1000 |
  | `agents/baseline_75k` | 828.5 | | 828.5 / 1000 |
  | `agents/baseline` | 905.5 | | 905.5 / 1000 |

- **Where it still loses** (its 108 losses in the 1,200 bot games): 56% against guess-sf-16x1000; 87% happen after ply 20. In 59% its king was already attacked by a piece it could not see; in 27% its king stepped onto a square attacked by a piece it could not see; in 8% it left its king attacked by a piece it could see. It never missed a chance to take the enemy king. Compared with `agents/turn_transformer` (152 losses), the king-steps-into-a-hidden-attack losses fell from 80 to 29
- **Hardware and GPU-hours:** everything ran on a personal Windows PC (RTX 3060 Ti, 16 CPU threads), not Sol. GPU: about 1.5 GPU-hours in total for the whole chain (training `move_history`, `move_history_branch` and `turn_transformer`, the learning loop's training steps and the rating games); the rest is CPU game play and Stockfish labelling, about 5 hours on 10-12 CPU workers
- **How to reproduce:** `bash reproduce.sh` from a fresh clone (see `REPRODUCE.md`). It rebuilds every stage: data for `move_history`, its training, the branching games, `turn_transformer`, `move_history_branch`, the value warm-up games and the learning loop, then copies the best round into `submission/`. The bots are not deterministic, so the rebuilt weights will differ from these
