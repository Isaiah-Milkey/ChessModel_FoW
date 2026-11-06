"""Baseline agent: a small CNN that imitates full-board Stockfish, playing from the fog.

It keeps a memory of enemy pieces it has seen, encodes the observation (features.py), scores
every move with the network (model.py) and plays the highest-scoring legal move. No search.

For its first few moves it instead samples from the network's move probabilities (seeded by
the game seed), so games vary while staying reproducible for a given seed.
"""
import json
import math
import os
import random
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import encode, legal_indices  # noqa: E402
from model import PolicyNet  # noqa: E402
from fow.memory import Memory  # noqa: E402

RANDOM_MOVES = 4     # own moves sampled at the start of each game
TEMPERATURE = 1.0    # softmax temperature for those moves (lower = closer to the top move)


class Agent:
    def __init__(self, agent_dir: str, device: str):
        cfg = json.load(open(os.path.join(agent_dir, "config.json")))
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.net = PolicyNet(cfg["in_planes"], cfg["channels"], cfg["blocks"])
        self.net.load_state_dict(load_file(os.path.join(agent_dir, "weights", "model.safetensors")))
        self.net.to(self.device).eval()

    def reset(self, color: str, seed: int) -> None:
        self.memory = Memory()
        self.rng = random.Random(seed)
        self.moves_played = 0

    @torch.no_grad()
    def act(self, obs: dict) -> str:
        self.memory.update(obs)
        x = torch.from_numpy(encode(obs, self.memory.last_seen)).float().unsqueeze(0).to(self.device)
        logits = self.net(x)[0].cpu()
        legal = legal_indices(obs)
        self.moves_played += 1
        if self.moves_played <= RANDOM_MOVES:
            idx = sorted(legal)
            scores = [logits[i].item() / TEMPERATURE for i in idx]
            top = max(scores)
            weights = [math.exp(s - top) for s in scores]
            return legal[self.rng.choices(idx, weights=weights)[0]]
        best = max(legal, key=lambda i: logits[i].item())
        return legal[best]
