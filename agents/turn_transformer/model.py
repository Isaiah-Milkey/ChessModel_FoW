"""move_history_branch's CNN plus a small transformer that reads the game turn by turn.

For every turn, the CNN (stem and residual blocks) reads that turn's 29 input planes. Each turn's
CNN features are squeezed into one vector (embed: conv 1x1 to 8 channels, 512 -> d_model), and a
causal transformer reads those vectors in order, so each turn sees itself and the turns before it
(at most `context` turns). Its output scales and shifts the current turn's CNN features (FiLM)
before the usual move head. The FiLM layer starts at zero, so an untrained transformer changes
nothing and the network starts as a plain CNN.

With use_context=False there is no transformer: the same CNN as move_history_branch (the control).
Dropout DROPOUT in the CNN (as move_history_branch) and in the transformer, training only.
Attention is written with plain matrix multiplications so the FLOP counter sees all of it.
"""
import math

import torch
import torch.nn as nn

DROPOUT = 0.10


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.drop = nn.Dropout(DROPOUT)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = self.drop(torch.relu(self.b1(self.c1(x))))
        return torch.relu(x + self.b2(self.c2(y)))


class TurnLayer(nn.Module):
    """Pre-norm transformer layer with causal self-attention over turns."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, z):                                    # z: [B, T, d]
        B, T, d = z.shape
        q, k, v = self.qkv(self.ln1(z)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // self.heads)       # [B, h, T, T]
        future = torch.ones(T, T, dtype=torch.bool, device=z.device).triu(1)
        att = self.drop(torch.softmax(att.masked_fill(future, float("-inf")), -1))
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, T, d)
        z = z + self.drop(self.proj(y))
        return z + self.drop(self.mlp(self.ln2(z)))


class PolicyNet(nn.Module):
    def __init__(self, in_planes=29, channels=64, blocks=4, d_model=128, layers=2, heads=4, context=64,
                 use_context=True):
        super().__init__()
        scale = torch.ones(in_planes, 1, 1)
        if in_planes > 22:
            scale[22] = 1 / 100                  # turn plane: features.PLY_CAP
        self.register_buffer("scale", scale, persistent=False)
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Conv2d(channels, 64, 1)   # channel = destination square
        self.use_context, self.context = use_context, context
        if use_context:
            self.embed = nn.Sequential(nn.Conv2d(channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                                       nn.Flatten(), nn.Linear(8 * 64, d_model))
            self.pos = nn.Parameter(torch.randn(context, d_model) * 0.02)
            self.layers = nn.ModuleList([TurnLayer(d_model, heads) for _ in range(layers)])
            self.ln = nn.LayerNorm(d_model)
            self.film = nn.Linear(d_model, 2 * channels)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def trunk(self, x):                          # x: [N, in_planes, 8, 8] -> [N, channels, 8, 8]
        return self.blocks(self.stem(x * self.scale))

    def turn_context(self, e):                   # e: [B, T, d_model], T <= context -> [B, T, d_model]
        z = e + self.pos[:e.shape[1]]
        for layer in self.layers:
            z = layer(z)
        return self.ln(z)

    def policy(self, h, c=None):                 # h: [N, channels, 8, 8], c: [N, d_model] or None
        if c is not None:
            g, b = self.film(c).chunk(2, -1)
            h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        out = self.head(self.drop(h))            # [N, to=64, 8, 8 (from)]
        return out.flatten(2).transpose(1, 2).reshape(h.shape[0], 64 * 64)  # index = from*64 + to

    def forward(self, x):                        # one turn without context (plain CNN)
        return self.policy(self.trunk(x))
