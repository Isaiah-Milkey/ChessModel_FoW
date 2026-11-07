# Model card: memory_age

- **Name:** imitation CNN with fading memory (residual CNN, no search), random openings
- **Core model:** `model.py` `PolicyNet(in_planes=22, channels=64, blocks=4)`; scores every (from, to) square pair. Same architecture and size as `agents/baseline_75k`; the only addition is a fixed, unsaved buffer that scales memory planes 13-18 from 0..255 to 0..1
- **Parameters:** 314,057 as counted by `python -m fow.params agents/memory_age`
- **Weight files and SHA-256:** `weights/model.safetensors`, `7e1f0ecf9247120ce400aa2bb32afb77929b13e8223d34cb2063c49279beeb70`
- **Input:** 22 planes; see `features.py`. As in the baseline, except that a remembered enemy piece is written as `max(1, round(255 * 0.85 ** age))` instead of 1, where age is the number of own turns since it was last seen (`memory.py`). It keeps 52% of its strength after 4 turns
- **Play:** the first 4 own moves are sampled from `softmax(logits / 1.0)` over legal moves using the game seed; after that, the top-scoring legal move. See `agent.py`
- **Training data:** the same 1,520,478 positions, labels and train/validation split as `agents/baseline_75k`, re-encoded with this agent's memory. Generated with `baselines/imitation/generate_data.py --games 100000 --workers 12 --out data/imitation_100k_age --seed 2 --agent-dir agents/memory_age`; only the 9 shards `baseline_75k` used (`shard_2_{0,1,2,3,4,7,8,10,11}.npz`) were kept. Part of shard 7's labels (23,498, from 63.8% of the way through) differed from the original run, so the original labels were used there
- **Training:** `baselines/imitation/train.py --data data/imitation_100k_age --agent-dir agents/memory_age --channels 64 --blocks 4 --epochs 16 --seed 0 --device cuda --save-best` (AdamW, one-cycle learning rate 2e-3, batch 256); the weights are from epoch 10, the best validation epoch
- **Training compute:** about 70 minutes of data generation on 12 CPU workers, 12 minutes of training on one RTX 3060 Ti
- **Validation match with teacher:** 42.0% of held-out positions (epoch 10)
- **Local rating:** 200 games each against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000, on three arena seeds:

  | Seed | vs greedy | vs guess-sf-4x500 | vs guess-sf-16x1000 | Rating (95%) | `baseline_75k`, same seed |
  |---|---:|---:|---:|---|---|
  | 4 | 175 | 175 | 148 | 1388 [1359, 1423] | 1338 [1296, 1372] |
  | 5 | 174 | 167 | 139 | 1351 [1318, 1385] | 1340 [1306, 1369] |
  | 6 | 169 | 172 | 142 | 1357 [1326, 1392] | 1348 [1310, 1374] |

  Head to head against `baseline_75k`: 510.5 / 1000 (a tie within noise).
