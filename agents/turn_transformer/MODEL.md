# Model card: turn_transformer

- **Name:** `move_history_branch`'s CNN plus a small causal transformer over the game's turns (no search), random openings
- **Core model:** `model.py` `PolicyNet(in_planes=29, channels=64, blocks=4, d_model=128, layers=2, heads=4, context=64)`. Per turn, the residual CNN of `agents/move_history_branch` (64 channels, 4 blocks) reads the 29 input planes. Each turn's CNN features are squeezed into one 128-number vector (conv 1x1 to 8 channels, batch norm, ReLU, 512 -> 128). A 2-layer pre-norm causal transformer (4 heads, 4x GELU MLP, learned position vectors) reads those vectors for the agent's last 64 turns at most, so each turn sees itself and the turns before it. Its output scales and shifts the current turn's CNN features (FiLM, initialised to zero) before the move head, which scores every (from, to) square pair. Dropout 0.1 in the CNN and the transformer, training only (`agent.py` calls `.eval()`). Attention is written with plain matrix multiplications so the FLOP counter sees all of it
- **Parameters:** 805,802 as counted by `python -m fow.params agents/turn_transformer` (804,624 trainable weights plus batch-norm statistics); under the 1M size floor, so the same size bonus as any smaller model
- **Weight files and SHA-256:** `weights/model.safetensors`, `9e1d3663aeb3d5b30e911f0e6b6de8cf5f3eb3635bd457631c6834127c089b80`
- **Input:** 29 planes per turn, exactly as `agents/move_history_branch` (`features.py`, `history.py`): the baseline's 22 planes (`fow.memory.Memory` for remembered pieces), plane 22 = ply number capped at 100, planes 23-28 = from and to squares of this agent's own last 3 moves, flipped for Black
- **State within a game:** `agent.py` keeps this game's turn vectors (the last 64) and clears them in `reset`, so nothing is carried between games
- **Play:** the first 4 own moves are sampled from `softmax(logits / 1.0)` over legal moves using the game seed; after that, the top-scoring legal move. See `agent.py`
- **Training data:** the same 54,000 branching games as `agents/move_history_branch` (`data/branch54k_s4`, recorded with `generate_data.py --record`; see that model card), replayed and encoded turn by turn: 108,000 sequences (one per side), 2,246,161 turns, of which 1,508,144 carry a Stockfish 19 label (5,000 nodes, full board). Unlabelled turns are context only. Command (about 1 minute):
  `python baselines/sequence/encode_seq.py --games data/branch54k_s4 --out data/branch54k_s4_seq`
- **Training:** `python baselines/sequence/train_seq.py --data data/branch54k_s4_seq --agent-dir agents/turn_transformer` (defaults: 16 epochs, batches of 48 chunks of up to 64 turns, cross-entropy on the labelled turns, AdamW with weight decay 1e-4, one-cycle learning rate 2e-3 for the CNN and 5e-4 for the transformer parts, 5% of games held out with seed 0); the weights are from epoch 11, the best held-out epoch
- **Training compute:** 19 minutes on one RTX 3060 Ti (about 0.3 GPU-hours), on a personal Windows PC, not Sol; plus the data generation of `move_history_branch` (78 minutes on 10 CPU workers)
- **Validation match with teacher:** 40.1% of the labelled turns of the held-out games (75,166 turns). The control (`--no-context`: the same CNN trained on exactly the same batches without the transformer) scores 39.3%
- **Validity:** `python -m fow.validate agents/turn_transformer --device cuda` prints VALID: mean 4.44e7, max 5.02e7 FLOPs per move, same moves on replay. Replaying recorded games through `agent.py` turn by turn gives the same move scores as the batch computation used in training (to within 1e-5)
- **Local rating:** 200 games each against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000 (the guess-sf bots do not play the same moves on every run):

  | Run | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) |
  |---|---:|---:|---:|---|
  | seed 6 | 181 | 189 | 167 | 1488 [1443, 1532] |
  | seed 7 | 183 | 182 | 146 | 1417 [1377, 1452] |
  | all 1,200 games | 364 / 400 | 371 / 400 | 313 / 400 | 1450 [1422, 1482] |

  On the same seeds, `move_history_branch` rated 1414 and the control 1377.
- **Head to head** (1,000 games per seed, seeds 6 and 7):

  | Opponent | Seed 6 | Seed 7 | Total |
  |---|---:|---:|---:|
  | control (same CNN, same batches, no transformer) | 528 | 559 | 1087 / 2000 (54.4%) |
  | `agents/move_history_branch` | 588.5 | 606 | 1194.5 / 2000 (59.7%) |
  | `move_history_branch` fine-tuned on its games with a value head (`runs/ft_vppo_l08`) | 549 | 589 | 1138 / 2000 (56.9%) |

- **Data checks:** trained on 25% and 50% of the training games (same held-out games and number of steps), it rated 1369 and 1393 and matched the teacher 37.7% and 39.0%. Adding 75,000 older games (`data/rec75k_s3`, played by `move_history` without branching) raised the match to 40.9% but lost 927 / 2000 head to head against this model, so it was not kept
- **How to reproduce:** not yet in `reproduce.sh`. It needs the data of `agents/move_history_branch` (see its card), then the two commands above
