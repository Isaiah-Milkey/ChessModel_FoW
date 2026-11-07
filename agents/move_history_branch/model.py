"""move_history's network with dropout: a small residual CNN policy, one score per (from, to) pair.

Dropout is applied inside each residual block (after the first convolution) and before the
head, during training only; agent.py calls .eval(), which turns it off for play.
The turn plane (22) arrives as 0..100 (features.PLY_CAP) and is scaled to 0..1 here.
"""
import torch
import torch.nn as nn

DROPOUT = 0.10   # chance each activation is zeroed during training


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


class PolicyNet(nn.Module):
    def __init__(self, in_planes=22, channels=64, blocks=4):
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

    def forward(self, x):                        # x: [B, in_planes, 8, 8]
        h = self.head(self.drop(self.blocks(self.stem(x * self.scale))))  # [B, to=64, 8, 8 (from)]
        return h.flatten(2).transpose(1, 2).reshape(x.shape[0], 64 * 64)  # index = from*64 + to
