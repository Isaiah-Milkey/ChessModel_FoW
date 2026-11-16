"""Train a value head: predicts the result of a Fog-of-War game (1 win, -1 loss for the agent)
from one of the agent's positions and the opponent it faces, using the game dataset from
play_games.py.

    python baselines/fow_games/value.py --data data/fow_games --init agents/move_history_branch --out runs/value_head

The head sits on the convolutional trunk of the --init policy network (its stem and residual
blocks): conv 1x1 to 32 channels, batch norm, ReLU, then 2048 -> 128, plus a learned vector for
the opponent (known during training, so the value may use it), ReLU, 128 -> 1 with tanh. With
--train-trunk the trunk is a trainable copy; otherwise it is frozen, so the value reads the
policy's own features. The policy network itself is never changed here.

It is used only during fine-tuning (finetune.py --value), to judge each move against how good
its position was, so it is not part of the agent and does not count toward its parameters.
Writes <out>/value.safetensors (trunk and head) and <out>/value.json. Uses the same held-out
games as finetune.py with the same --val and --seed, and reports the held-out error against
the simple baseline (the agent's average result against that opponent with that colour).
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play_games import load_net  # noqa: E402


class ValueNet(nn.Module):
    def __init__(self, policy, channels, n_opponents):
        super().__init__()
        self.register_buffer("scale", policy.scale.clone(), persistent=False)
        self.stem = copy.deepcopy(policy.stem)
        self.blocks = copy.deepcopy(policy.blocks)
        self.head = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(32 * 64, 128))
        self.opponent = nn.Embedding(n_opponents, 128)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x, opp):
        return self.out(self.head(self.blocks(self.stem(x * self.scale))) + self.opponent(opp))[:, 0]


def held_out_games(games, val, seed):
    """The same held-out games as finetune.py with the same --val and --seed."""
    return np.random.RandomState(seed).permutation(games)[:int(val * len(games))]


def load_value(path, init, device):
    """ValueNet from <path>/value.safetensors, on top of the --init policy's architecture."""
    cfg = json.load(open(os.path.join(path, "value.json")))
    net = ValueNet(load_net(init), cfg["channels"], cfg["n_opponents"])
    net.load_state_dict(load_file(os.path.join(path, "value.safetensors")))
    return net.to(device).eval()


def predict(net, x, opp, device, bs=4096):
    out = np.empty(len(x), np.float32)
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out[i:i + bs] = net(torch.from_numpy(x[i:i + bs]).float().to(device),
                                torch.from_numpy(opp[i:i + bs].astype(np.int64)).to(device)).cpu().numpy()
    return out


def main():
    from finetune import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fow_games")
    ap.add_argument("--init", default="agents/move_history_branch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-trunk", action="store_true", help="train a copy of the trunk too (default: frozen)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    d = load(args.data)
    games = np.unique(d["game"])
    is_val = np.isin(d["game"], held_out_games(games, args.val, args.seed))
    tr_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    r = d["result"].astype(np.float32)
    group = d["opponent"].astype(np.int64) * 2 + (d["ply"] % 2)
    b = np.bincount(group[tr_idx], r[tr_idx]) / np.bincount(group[tr_idx])
    base_mse = np.mean((r[val_idx] - b[group[val_idx]]) ** 2)

    dev = torch.device(args.device)
    cfg = json.load(open(os.path.join(args.init, "config.json")))
    n_opp = int(d["opponent"].max()) + 1
    net = ValueNet(load_net(args.init), cfg["channels"], n_opp).to(dev)
    if not args.train_trunk:
        for m in (net.stem, net.blocks):
            for p in m.parameters():
                p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"{len(tr_idx):,} training / {len(val_idx):,} held-out turns; {sum(p.numel() for p in params):,} "
          f"trainable parameters; held-out MSE of the opponent-and-colour baseline {base_mse:.4f}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(tr_idx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    def set_mode():
        net.train()
        if not args.train_trunk:          # frozen trunk keeps its batch-norm statistics
            net.stem.eval()
            net.blocks.eval()

    rng = np.random.RandomState(args.seed + 1)
    for ep in range(args.epochs):
        t = time.time()
        set_mode()
        rng.shuffle(tr_idx)
        tot = 0.0
        for i in range(0, len(tr_idx), args.batch):
            idx = tr_idx[i:i + args.batch]
            v = net(torch.from_numpy(d["x"][idx]).float().to(dev),
                    torch.from_numpy(d["opponent"][idx].astype(np.int64)).to(dev))
            loss = ((v - torch.from_numpy(r[idx]).to(dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        net.eval()
        v = predict(net, d["x"][val_idx], d["opponent"][val_idx], dev)
        rv, pv = r[val_idx], d["ply"][val_idx]
        mse = np.mean((v - rv) ** 2)
        by_stage = ", ".join(f"{lo}-{hi}: {np.mean((v - rv)[(pv >= lo) & (pv < hi)] ** 2):.3f}"
                             for lo, hi in ((0, 20), (20, 40), (40, 400)))
        print(f"epoch {ep + 1}: train MSE {tot / len(tr_idx):.4f} | held out: MSE {mse:.4f} (baseline {base_mse:.4f}), "
              f"sign right {np.mean(np.sign(v) == rv):.1%}, MSE by ply {by_stage} ({time.time() - t:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
              os.path.join(args.out, "value.safetensors"))
    json.dump({**vars(args), "channels": cfg["channels"], "n_opponents": n_opp, "heldout_mse": float(mse), "baseline_mse": float(base_mse)},
              open(os.path.join(args.out, "value.json"), "w"), indent=1)
    print(f"saved to {args.out}/value.safetensors")


if __name__ == "__main__":
    main()
