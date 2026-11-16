"""Train the turn-transformer policy (agents/turn_transformer/model.py) on turn sequences from
encode_seq.py, imitating the Stockfish labels, and write it into an agent directory.

    python baselines/sequence/train_seq.py --data data/branch54k_s4_seq --agent-dir agents/turn_transformer
    python baselines/sequence/train_seq.py --data data/branch54k_s4_seq --agent-dir runs/seq_control --no-context

Each side's sequence of turns is cut into chunks of at most --context turns (the agent also looks
back at most that many turns). A batch holds --batch chunks: the CNN reads every turn, the
transformer reads each chunk in order, and the loss is the cross-entropy with the teacher move on
the labelled turns only. --no-context trains the same CNN without the transformer on exactly the
same batches (it skips the unlabelled turns, which only serve as context): the control.

Validation holds out whole games (both sides' sequences): --val of them, chosen with --seed.
--games-frac trains on a random part of the remaining games (same held-out games), to measure how
much more data helps; give it proportionally more --epochs to keep the number of steps equal.
--extra-data adds more sequence dirs for training only, so the held-out games (from --data) stay
the same and the match numbers stay comparable.
The agent directory must contain model.py; this script writes config.json and
weights/model.safetensors there (and copies the agent code there if it is missing).
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode_seq import unpack  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from guard import Guard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, "agents", "turn_transformer")


def load(dirs):
    """All shards of all dirs joined: per-turn arrays, a legal-mask row for each labelled turn,
    sequences (s_game is unique across shards; s_dir is the index of the sequence's directory)."""
    bits, ply, label, legal, lrow, s_lo, s_hi, s_game, s_dir = [], [], [], [], [], [], [], [], []
    n_rows = n_leg = 0
    paths = [(i, p) for i, dr in enumerate(dirs) for p in sorted(glob.glob(os.path.join(dr, "seq_*.npz")))]
    for k, (i, path) in enumerate(paths):
        d = np.load(path)
        lab = d["label"]
        lr = np.full(len(lab), -1, np.int64)
        lr[lab >= 0] = n_leg + np.arange(int((lab >= 0).sum()))
        bits.append(d["bits"]); ply.append(d["ply"]); label.append(lab); legal.append(d["legal"]); lrow.append(lr)
        off = d["seq_off"]
        s_lo.append(off[:-1] + n_rows); s_hi.append(off[1:] + n_rows)
        s_game.append(d["seq_game"].astype(np.int64) + k * 10_000_000)
        s_dir.append(np.full(len(off) - 1, i))
        n_rows += len(lab); n_leg += len(d["legal"])
    cat = np.concatenate
    return dict(bits=cat(bits), ply=cat(ply), label=cat(label).astype(np.int64), legal=cat(legal),
                lrow=cat(lrow), s_lo=cat(s_lo), s_hi=cat(s_hi), s_game=cat(s_game), s_dir=cat(s_dir))


def chunks(d, seqs, context):
    """(first row, length) of every chunk of at most `context` turns with at least one label."""
    out = []
    for s in seqs:
        for a in range(d["s_lo"][s], d["s_hi"][s], context):
            b = min(a + context, d["s_hi"][s])
            if (d["label"][a:b] >= 0).any():
                out.append((a, b - a))
    return np.array(out, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/branch54k_s4_seq")
    ap.add_argument("--extra-data", default="", help="comma-separated sequence dirs used for training only")
    ap.add_argument("--agent-dir", default="agents/turn_transformer")
    ap.add_argument("--no-context", action="store_true", help="no transformer: the plain CNN (control)")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--context", type=int, default=64, help="most turns the transformer looks back over")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=48, help="chunks per batch (about 20 turns each)")
    ap.add_argument("--lr", type=float, default=2e-3, help="peak learning rate of the CNN")
    ap.add_argument("--tf-lr", type=float, default=5e-4, help="peak learning rate of the transformer parts")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val", type=float, default=0.05, help="fraction of games held out")
    ap.add_argument("--games-frac", type=float, default=1.0,
                    help="train on this fraction of the training games (held-out games unchanged)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="stop each epoch after this many batches (for timing)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    guard = Guard("train_seq")                            # one heavy job at a time; pauses while overloaded
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    extra = [p for p in args.extra_data.split(",") if p]
    d = load([args.data] + extra)
    main_seq = d["s_dir"] == 0
    games = np.unique(d["s_game"][main_seq])            # held-out games come from --data only
    held_games = rng.permutation(games)[:int(args.val * len(games))]
    held = np.isin(d["s_game"], held_games)
    train = ~held
    if args.games_frac < 1:                              # same held-out games, fewer training games
        rest = np.setdiff1d(games, held_games)
        keep = np.random.RandomState(args.seed + 1).permutation(rest)[:int(args.games_frac * len(rest))]
        train = np.isin(d["s_game"], keep) | ~main_seq
        print(f"training on {len(keep):,} of {len(rest):,} training games of --data (--games-frac {args.games_frac})")
    if extra:
        print(f"plus {len(np.unique(d['s_game'][~main_seq])):,} games from {', '.join(extra)} (training only)")
    tr_chunks = chunks(d, np.flatnonzero(train), args.context)
    va_chunks = chunks(d, np.flatnonzero(held), args.context)
    n_lab = lambda cs: sum(int((d["label"][a:a + n] >= 0).sum()) for a, n in cs)
    print(f"{len(d['label']):,} turns, {len(d['s_lo']):,} sequences from {len(games):,} games; "
          f"{len(tr_chunks):,} training chunks ({n_lab(tr_chunks):,} labelled turns), "
          f"{len(va_chunks):,} held-out chunks ({n_lab(va_chunks):,} labelled turns)", flush=True)

    os.makedirs(os.path.join(args.agent_dir, "weights"), exist_ok=True)
    for f in ("agent.py", "features.py", "history.py", "model.py"):
        if not os.path.exists(os.path.join(args.agent_dir, f)):
            shutil.copy(os.path.join(CODE, f), args.agent_dir)
    spec = importlib.util.spec_from_file_location("agent_model", os.path.join(args.agent_dir, "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = dict(in_planes=29, channels=args.channels, blocks=args.blocks, d_model=args.d_model, layers=args.layers,
               heads=args.heads, context=args.context, use_context=not args.no_context)
    dev = torch.device(args.device)
    net = m.PolicyNet(**cfg).to(dev)
    cnn = [p for n_, p in net.named_parameters() if n_.split(".")[0] in ("stem", "blocks", "head")]
    tf = [p for n_, p in net.named_parameters() if n_.split(".")[0] not in ("stem", "blocks", "head")]
    print(f"model: {sum(p.numel() for p in net.parameters()):,} parameters "
          f"({sum(p.numel() for p in tf):,} in the transformer parts)", flush=True)
    groups = [{"params": cnn, "lr": args.lr}] + ([{"params": tf, "lr": args.tf_lr}] if tf else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    per_epoch = math.ceil(len(tr_chunks) / args.batch)
    if args.max_batches:
        per_epoch = min(per_epoch, args.max_batches)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=args.epochs * per_epoch, pct_start=0.1)

    def run(cs):
        """Logits of the labelled turns of chunks cs, and their labels."""
        lens = cs[:, 1]
        rows = np.concatenate([np.arange(a, a + n) for a, n in cs])
        lab = d["label"][rows] >= 0
        if args.no_context:                               # the control needs only the labelled turns
            rows, lab = rows[lab], np.ones(lab.sum(), bool)
        x = torch.from_numpy(unpack(d["bits"][rows], d["ply"][rows])).float().to(dev)
        h = net.trunk(x)
        lab_t = torch.from_numpy(lab).to(dev)
        c = None
        if not args.no_context:
            e = net.embed(h)
            bi = torch.from_numpy(np.repeat(np.arange(len(cs)), lens)).to(dev)
            ti = torch.from_numpy(np.concatenate([np.arange(n) for n in lens])).to(dev)
            E = torch.zeros(len(cs), int(lens.max()), e.shape[1], device=dev, dtype=e.dtype)
            E[bi, ti] = e
            c = net.turn_context(E)[bi, ti][lab_t]
        logits = net.policy(h[lab_t], c)
        lr_ = d["lrow"][rows[lab]]
        mb = torch.from_numpy(np.unpackbits(d["legal"][lr_], axis=1)).bool().to(dev)
        y = torch.from_numpy(d["label"][rows[lab]]).to(dev)
        return logits.masked_fill(~mb, -1e9), y

    best = (-1.0, 0, None)
    for ep in range(args.epochs):
        net.train()
        t = time.time()
        order = rng.permutation(len(tr_chunks))[:per_epoch * args.batch]
        tot = cnt = 0.0
        for i in range(0, len(order), args.batch):
            if (i // args.batch) % 100 == 0:
                guard.wait_ok()
            logits, y = run(tr_chunks[order[i:i + args.batch]])
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            cnt += len(y)
        net.eval()
        hits = n = 0
        with torch.no_grad():
            for i in range(0, len(va_chunks), 128):
                logits, y = run(va_chunks[i:i + 128])
                hits += (logits.argmax(1) == y).sum().item()
                n += len(y)
        acc = hits / n
        print(f"epoch {ep + 1}: train loss {tot / cnt:.3f}, held-out games: match with teacher {acc:.3f} "
              f"({time.time() - t:.0f}s)", flush=True)
        if acc > best[0]:
            best = (acc, ep + 1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

    print(f"keeping epoch {best[1]} (held-out match {best[0]:.3f})")
    save_file({k: v.contiguous() for k, v in best[2].items()}, os.path.join(args.agent_dir, "weights", "model.safetensors"))
    json.dump(cfg, open(os.path.join(args.agent_dir, "config.json"), "w"), indent=1)
    print(f"saved to {args.agent_dir}/weights/model.safetensors")


if __name__ == "__main__":
    main()
