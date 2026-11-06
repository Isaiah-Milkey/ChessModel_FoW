# Model card: baseline_4x

- **Name:** 4x-parameter imitation baseline (residual CNN, no search), random openings
- **Core model:** `model.py` `PolicyNet(in_planes=22, channels=128, blocks=4)`; scores every (from, to) square pair. Same architecture as `agents/baseline`, twice the width (about 4x the parameters)
- **Parameters:** 1,217,865 as counted by `python -m fow.params agents/baseline_4x`
- **Weight files and SHA-256:** `weights/model.safetensors`, `61e8f63624b75c846baeacf46244bed8aaabdc2cd974ac28b3b83e1f19fae89a`
- **Input:** 22 planes; see `features.py`
- **Play:** the first 4 own moves are sampled from `softmax(logits / 1.0)` over legal moves using the game seed; after that, the top-scoring legal move. See `agent.py`
- **Training data:** 405,888 positions from 20,000 games between reference bots, each labelled with full-board Stockfish 19 at 5,000 nodes (`baselines/imitation/generate_data.py --games 20000 --workers 12 --out data/imitation_20k --seed 1`; the worker count changes which games are played)
- **Training:** `baselines/imitation/train.py --data data/imitation_20k --agent-dir agents/baseline_4x --channels 128 --blocks 4 --epochs 8 --seed 0 --device cuda` (AdamW, one-cycle learning rate 2e-3, batch 256)
- **Training compute:** about 14 minutes of data generation on 12 CPU workers, 3.5 minutes of training on one RTX 3060 Ti
- **Validation match with teacher:** 40.2% of held-out positions after epoch 8 (peak 41.0% at epoch 5)
- **Local rating:** 1289 [1242, 1332] from 200 games each against bot:greedy and bot:guess-sf-4x500
