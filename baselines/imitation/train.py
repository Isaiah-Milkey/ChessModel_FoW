"""Train the baseline policy on data from generate_data.py and write it into an agent directory.

    python baselines/imitation/train.py --data data/imitation --agent-dir submission --epochs 8

The agent directory must contain model.py (the network); this script writes
config.json and weights/model.safetensors there.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def load_model_class(agent_dir):
    spec = importlib.util.spec_from_file_location("agent_model", os.path.join(agent_dir, "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.PolicyNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imitation")
    ap.add_argument("--agent-dir", default="submission")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-best", action="store_true",
                    help="keep the weights from the epoch with the best validation match, not the last")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shards = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    data = [np.load(s) for s in shards]
    x = np.concatenate([d["x"] for d in data])
    y = np.concatenate([d["y"] for d in data]).astype(np.int64)
    legal = np.concatenate([d["legal"] for d in data])
    n = len(y)
    perm = np.random.permutation(n)
    n_val = max(1, int(args.val * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"{n} positions from {len(shards)} shards; {len(tr_idx)} train / {n_val} validation")

    dev = torch.device(args.device)
    in_planes = x.shape[1]
    net = load_model_class(args.agent_dir)(in_planes, args.channels, args.blocks).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"model: {n_params:,} parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(tr_idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    def batch(idx):
        xb = torch.from_numpy(x[idx]).float().to(dev)
        yb = torch.from_numpy(y[idx]).to(dev)
        mb = torch.from_numpy(np.unpackbits(legal[idx], axis=1)).bool().to(dev)
        return xb, yb, mb

    def masked_logits(xb, mb):
        return net(xb).masked_fill(~mb, -1e9)

    best_acc, best_ep, best_state = -1.0, 0, None
    for ep in range(args.epochs):
        net.train()
        t = time.time()
        np.random.shuffle(tr_idx)
        tot = 0.0
        for i in range(0, len(tr_idx), args.batch):
            xb, yb, mb = batch(tr_idx[i:i + args.batch])
            loss = F.cross_entropy(masked_logits(xb, mb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
        net.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n_val, 1024):
                xb, yb, mb = batch(val_idx[i:i + 1024])
                correct += (masked_logits(xb, mb).argmax(1) == yb).sum().item()
        print(f"epoch {ep + 1}: train loss {tot / len(tr_idx):.3f}, "
              f"validation match with teacher {correct / n_val:.3f} ({time.time() - t:.0f}s)", flush=True)
        if correct / n_val > best_acc:
            best_acc, best_ep = correct / n_val, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    state = best_state if args.save_best else net.state_dict()
    if args.save_best:
        print(f"keeping epoch {best_ep} (validation match {best_acc:.3f})")
    os.makedirs(os.path.join(args.agent_dir, "weights"), exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()},
              os.path.join(args.agent_dir, "weights", "model.safetensors"))
    json.dump({"in_planes": in_planes, "channels": args.channels, "blocks": args.blocks},
              open(os.path.join(args.agent_dir, "config.json"), "w"), indent=1)
    print(f"saved to {args.agent_dir}/weights/model.safetensors")


if __name__ == "__main__":
    main()
