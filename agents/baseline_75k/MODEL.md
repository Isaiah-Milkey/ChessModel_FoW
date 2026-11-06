# Model card: baseline_75k

- **Name:** imitation baseline trained on about 19x more data (residual CNN, no search), random openings
- **Core model:** `model.py` `PolicyNet(in_planes=22, channels=64, blocks=4)`; scores every (from, to) square pair. Same architecture and size as `agents/baseline`
- **Parameters:** 314,057 as counted by `python -m fow.params agents/baseline_75k`
- **Weight files and SHA-256:** `weights/model.safetensors`, `9447ef379133069a6be4b3174b7740e9864cf75dc2ffa39392258a4d7222cc2b`
- **Input:** 22 planes; see `features.py`
- **Play:** the first 4 own moves are sampled from `softmax(logits / 1.0)` over legal moves using the game seed; after that, the top-scoring legal move. See `agent.py`
- **Training data:** 1,520,478 positions from about 75,000 games between reference bots, each labelled with full-board Stockfish 19 at 5,000 nodes. Generated with `baselines/imitation/generate_data.py --games 100000 --workers 12 --out data/imitation_100k --seed 2`; workers 5, 6 and 9 died silently, so only the other 9 of 12 shards (`shard_2_{0,1,2,3,4,7,8,10,11}.npz`) were used
- **Training:** `baselines/imitation/train.py --data data/imitation_100k --agent-dir agents/baseline_75k --channels 64 --blocks 4 --epochs 16 --seed 0 --device cuda --save-best` (AdamW, one-cycle learning rate 2e-3, batch 256); the weights are from epoch 12, the best validation epoch
- **Training compute:** about 70 minutes of data generation on 12 CPU workers, 12 minutes of training on one RTX 3060 Ti
- **Validation match with teacher:** 42.1% of held-out positions (epoch 12)
- **Local rating:** 1347 [1298, 1402] from 200 games each against bot:greedy (176 / 200) and bot:guess-sf-4x500 (168 / 200)
