"""Turn-transformer agent: move_history_branch's CNN plus a transformer over the game's turns.

Every turn it encodes the observation as move_history does (features.py, history.py), runs the
CNN, adds this turn's summary vector to the list of this game's turn vectors (the last `context`
of them), and lets the transformer read that list to adjust the move scores (model.py). The list
is cleared in reset, so nothing is carried between games. With use_context false in config.json
it is the plain CNN.

For its first few moves it samples from the network's move probabilities (seeded by the game
seed), so games vary while staying reproducible for a given seed; after that it plays the
top-scoring legal move. No search.
"""
import json
import math
import os
import random
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import encode_from_memory, legal_indices, make_memory  # noqa: E402
from model import PolicyNet  # noqa: E402

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
        self.net = PolicyNet(**cfg)
        self.net.load_state_dict(load_file(os.path.join(agent_dir, "weights", "model.safetensors")))
        self.net.to(self.device).eval()

    def reset(self, color: str, seed: int) -> None:
        self.memory = make_memory()
        self.rng = random.Random(seed)
        self.moves_played = 0
        self.turns = []                          # this game's turn vectors, oldest first

    @torch.no_grad()
    def act(self, obs: dict) -> str:
        self.memory.update(obs)
        x = torch.from_numpy(encode_from_memory(obs, self.memory)).float().unsqueeze(0).to(self.device)
        h = self.net.trunk(x)
        c = None
        if self.net.use_context:
            self.turns = (self.turns + [self.net.embed(h)])[-self.net.context:]
            c = self.net.turn_context(torch.cat(self.turns).unsqueeze(0))[0, -1:]
        logits = self.net.policy(h, c)[0].cpu()
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
