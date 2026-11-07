"""Train the baseline policy on data from generate_data.py and write it into an agent directory.

    python baselines/imitation/train.py --data data/imitation --agent-dir submission --epochs 8

The agent directory must contain model.py (the network); this script writes
config.json and weights/model.safetensors there.

Two options deal with positions that look identical to the agent (the same input planes):
  --split view   holds out whole views for validation, so no validation position also appears in
                 training (the default split is by position, and common openings appear in both).
  --merge        (implies --split view) trains on each distinct view once, with the share of
                 teacher moves given for it as a soft target, instead of once per copy.
With --split view, validation also reports the match per distinct view, each view counted once.
"""
import argparse
import glob
import hashlib
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


def view_ids(x):
    """An id per row of x; rows with identical input planes get the same id."""
    ids, seen = np.empty(len(x), np.int64), {}
    for i, row in enumerate(x):
        ids[i] = seen.setdefault(hashlib.blake2b(row.tobytes(), digest_size=16).digest(), len(seen))
    return ids, len(seen)


def view_split(ids, n_views, val, seed):
    """(validation positions, training positions), holding out a random fraction of the views."""
    held = np.zeros(n_views, bool)
    held[np.random.RandomState(seed).permutation(n_views)[:max(1, int(val * n_views))]] = True
    is_val = held[ids]
    return np.flatnonzero(is_val), np.flatnonzero(~is_val)


def merge_views(ids, y, legal, tr_idx):
    """For the training positions: one representative position per distinct view, the union of
    its legal masks, and its teacher moves with their shares (CSR: off, lab, share)."""
    v, lab = ids[tr_idx], y[tr_idx]
    order = np.lexsort((lab, v))
    v, lab, pos = v[order], lab[order], tr_idx[order]
    first = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])          # first position of each view
    rep = pos[first]
    union = np.bitwise_or.reduceat(legal[pos], first, axis=0)
    pair = np.flatnonzero(np.r_[True, (v[1:] != v[:-1]) | (lab[1:] != lab[:-1])])  # (view, move) runs
    cnt = np.diff(np.r_[pair, len(v)])
    pair_view = np.searchsorted(first, pair, side="right") - 1
    total = np.bincount(pair_view, weights=cnt)
    off = np.r_[np.searchsorted(pair_view, np.arange(len(first))), len(pair)]
    return rep, union, off, lab[pair].astype(np.int64), (cnt / total[pair_view]).astype(np.float32)


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
    ap.add_argument("--split", choices=["position", "view"], default="position",
                    help="hold out random positions (default) or whole distinct views for validation")
    ap.add_argument("--merge", action="store_true",
                    help="train on each distinct view once, targeting the share of each teacher move "
                         "(implies --split view)")
    args = ap.parse_args()
    if args.merge:
        args.split = "view"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shards = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    data = [np.load(s) for s in shards]
    x = np.concatenate([d["x"] for d in data])
    y = np.concatenate([d["y"] for d in data]).astype(np.int64)
    legal = np.concatenate([d["legal"] for d in data])
    n = len(y)
    if args.split == "position":
        perm = np.random.permutation(n)
        n_val = max(1, int(args.val * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        val_views = None
    else:
        ids, n_views = view_ids(x)
        val_idx, tr_idx = view_split(ids, n_views, args.val, args.seed)
        n_val = len(val_idx)
        val_views = np.unique(ids[val_idx], return_inverse=True)[1]   # 0..k-1 per validation position
        print(f"{n_views:,} distinct views; {val_views.max() + 1:,} held out for validation")
    print(f"{n} positions from {len(shards)} shards; {len(tr_idx)} train / {n_val} validation")

    if args.merge:
        rep, union, off, soft_lab, soft_w = merge_views(ids, y, legal, tr_idx)
        examples = np.arange(len(rep))
        print(f"merged: {len(rep):,} training views from {len(tr_idx):,} positions; "
              f"{np.mean(np.diff(off) > 1):.1%} of them have more than one teacher move")
    else:
        examples = tr_idx

    dev = torch.device(args.device)
    in_planes = x.shape[1]
    net = load_model_class(args.agent_dir)(in_planes, args.channels, args.blocks).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"model: {n_params:,} parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(examples) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    def batch(idx):
        xb = torch.from_numpy(x[idx]).float().to(dev)
        yb = torch.from_numpy(y[idx]).to(dev)
        mb = torch.from_numpy(np.unpackbits(legal[idx], axis=1)).bool().to(dev)
        return xb, yb, mb

    def merged_batch(e):
        """Inputs, soft targets [B, 4096] and legal masks for merged training views e."""
        xb = torch.from_numpy(x[rep[e]]).float().to(dev)
        mb = torch.from_numpy(np.unpackbits(union[e], axis=1)).bool().to(dev)
        lens = off[e + 1] - off[e]
        rows = np.repeat(np.arange(len(e)), lens)
        k = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(off[e], lens)
        tb = torch.zeros(len(e), mb.shape[1], device=dev)
        tb[torch.from_numpy(rows).to(dev), torch.from_numpy(soft_lab[k]).to(dev)] = torch.from_numpy(soft_w[k]).to(dev)
        return xb, tb, mb

    def masked_logits(xb, mb):
        return net(xb).masked_fill(~mb, -1e9)

    best_acc, best_ep, best_state = -1.0, 0, None
    for ep in range(args.epochs):
        net.train()
        t = time.time()
        np.random.shuffle(examples)
        tot = 0.0
        for i in range(0, len(examples), args.batch):
            if args.merge:
                xb, tb, mb = merged_batch(examples[i:i + args.batch])
                loss = -(tb * F.log_softmax(masked_logits(xb, mb), 1)).sum(1).mean()
                nb = len(tb)
            else:
                xb, yb, mb = batch(examples[i:i + args.batch])
                loss = F.cross_entropy(masked_logits(xb, mb), yb)
                nb = len(yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * nb
        net.eval()
        hits = np.zeros(n_val, bool)
        with torch.no_grad():
            for i in range(0, n_val, 1024):
                xb, yb, mb = batch(val_idx[i:i + 1024])
                hits[i:i + 1024] = (masked_logits(xb, mb).argmax(1) == yb).cpu().numpy()
        acc = hits.mean()
        per_view = ""
        if val_views is not None:
            k = val_views.max() + 1
            per_view = f", per view {np.mean(np.bincount(val_views, hits, k) / np.bincount(val_views, None, k)):.3f}"
        print(f"epoch {ep + 1}: train loss {tot / len(examples):.3f}, "
              f"validation match with teacher {acc:.3f}{per_view} ({time.time() - t:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc, best_ep = acc, ep + 1
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
