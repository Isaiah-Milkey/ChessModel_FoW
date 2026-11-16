"""Fine-tune a trained agent on its own Fog-of-War games (from play_games.py), learning from how
each game ended instead of from a teacher.

    python baselines/fow_games/finetune.py --data data/fow_games --init agents/move_history_branch --out runs/ft_pg

Copies the code of --init into --out, starts from its weights, and writes config.json and
weights/model.safetensors into --out.

Every recorded agent move gets an advantage
    A = (result - b[opponent, colour]) * gamma ** (agent turns left in the game)
where result is 1 / 0 / -1 for the agent and b is the agent's average result against that
opponent with that colour (a baseline, so wins against a strong opponent count for more than
wins against a weak one). With gamma < 1, moves near the end of the game get more of the credit.
With --value (a value head from value.py), moves are instead judged against the predicted
result of their position: generalized advantage estimation, A_t = sum_k (gamma * lam)^k *
delta_(t+k), delta_t = gamma * V(next turn) - V(this turn), with the game result after the
last turn.

Loss (--method):
  pg    policy gradient:        -mean(A * log p(move))                 (moves in lost games are pushed down;
                                unbounded, so it diverges unless the KL weight is large)
  ppo   clipped policy gradient: -mean(min(rho * A, clip(rho, 1 - clip, 1 + clip) * A)),
                                rho = p(move) / p_start(move): stops once a move's probability has
                                moved by more than the clip fraction from the starting model
  awr   advantage-weighted:     -sum(w * log p(move)) / sum(w), w = exp(A / beta) (only pulls moves up)
plus kl * KL(start || current) over the legal moves of the same positions, which keeps the
policy close to the one it started from.

The network stays in eval mode (no dropout, batch-norm statistics frozen); only the weights
change. 5% of the games are held out: each epoch reports, on those games, how far the policy has
moved (KL, share of positions whose top move changed) and how the log-probability of the moves
played changed in won and in lost games.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play_games import load_net  # noqa: E402


def load(data):
    cols = ("x", "move", "legal", "result", "opponent", "plies_left", "game", "ply")
    parts = {k: [] for k in cols}
    for s in sorted(glob.glob(os.path.join(data, "shard_*.npz"))):
        d = np.load(s)
        for k in cols:
            parts[k].append(d[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fow_games")
    ap.add_argument("--init", default="agents/move_history_branch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["pg", "ppo", "awr"], default="ppo")
    ap.add_argument("--clip", type=float, default=0.2, help="ppo: largest relative change of p(move)")
    ap.add_argument("--gamma", type=float, default=1.0, help="credit decay per agent turn before the end")
    ap.add_argument("--beta", type=float, default=0.5, help="awr temperature")
    ap.add_argument("--kl", type=float, default=0.5, help="weight of KL(start || current)")
    ap.add_argument("--value", default=None,
                    help="directory from value.py: judge moves by the value head instead of the opponent baseline")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="with --value: 1 = result minus the position's value; lower = more weight on the "
                         "change in value over the next turns (GAE lambda)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--val", type=float, default=0.05, help="fraction of games held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    d = load(args.data)
    n = len(d["move"])
    games = np.unique(d["game"])
    held = rng.permutation(games)[:int(args.val * len(games))]
    is_val = np.isin(d["game"], held)
    tr_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    colour = (d["ply"] % 2).astype(np.int64)                  # 0 white, 1 black
    group = d["opponent"].astype(np.int64) * 2 + colour
    r = d["result"].astype(np.float32)
    b = np.bincount(group[tr_idx], r[tr_idx], minlength=group.max() + 1) / np.maximum(
        np.bincount(group[tr_idx], minlength=group.max() + 1), 1)
    adv = (r - b[group]) * args.gamma ** (d["plies_left"] // 2).astype(np.float32)
    print(f"{n:,} turns, {len(games):,} games ({len(held):,} held out); baseline by opponent and colour: "
          f"{np.round(b, 3).tolist()}; advantage mean |A| {np.abs(adv[tr_idx]).mean():.3f}", flush=True)

    os.makedirs(os.path.join(args.out, "weights"), exist_ok=True)
    for f in glob.glob(os.path.join(args.init, "*.py")) + [os.path.join(args.init, "config.json")]:
        shutil.copy(f, args.out)
    if not os.path.exists(os.path.join(args.out, "MODEL.md")):
        open(os.path.join(args.out, "MODEL.md"), "w").write(f"# placeholder: {args.init} fine-tuned on {args.data}\n")
    dev = torch.device(args.device)
    net = load_net(args.init).to(dev).eval()
    net0 = load_net(args.init).to(dev).eval()
    for p in net0.parameters():
        p.requires_grad_(False)
    if args.value:
        from value import load_value, predict
        V = predict(load_value(args.value, args.init, dev), d["x"], d["opponent"], dev)
        same_next = np.r_[d["game"][1:] == d["game"][:-1], False]       # next row is this game's next turn
        v_next = np.where(same_next, np.r_[V[1:], 0.0], r)                # the result after the last turn
        delta = args.gamma * v_next - V
        adv = np.zeros(n, np.float32)
        carry = 0.0
        for t in range(n - 1, -1, -1):
            carry = delta[t] + (args.gamma * args.lam * carry if same_next[t] else 0.0)
            adv[t] = carry
        print(f"value advantages (lambda {args.lam}): mean |A| {np.abs(adv[tr_idx]).mean():.3f}; "
              f"held-out value MSE {np.mean((V[val_idx] - r[val_idx]) ** 2):.4f}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    def batch(idx):
        xb = torch.from_numpy(d["x"][idx]).float().to(dev)
        mb = torch.from_numpy(np.unpackbits(d["legal"][idx], axis=1)).bool().to(dev)
        ab = torch.from_numpy(d["move"][idx].astype(np.int64)).to(dev)
        return xb, mb, ab

    def logp(model, xb, mb):
        return F.log_softmax(model(xb).masked_fill(~mb, -1e9), 1)

    def evaluate():
        kl, changed, dlp = 0.0, 0, np.zeros(len(val_idx), np.float32)
        with torch.no_grad():
            for i in range(0, len(val_idx), 2048):
                xb, mb, ab = batch(val_idx[i:i + 2048])
                lp, lp0 = logp(net, xb, mb), logp(net0, xb, mb)
                kl += torch.where(mb, lp0.exp() * (lp0 - lp), 0.0).sum().item()
                changed += (lp.argmax(1) != lp0.argmax(1)).sum().item()
                rows = torch.arange(len(ab), device=dev)
                dlp[i:i + 2048] = (lp[rows, ab] - lp0[rows, ab]).cpu().numpy()
        rv = d["result"][val_idx]
        return kl / len(val_idx), changed / len(val_idx), dlp[rv == 1].mean(), dlp[rv == -1].mean()

    for ep in range(args.epochs):
        t = time.time()
        rng.shuffle(tr_idx)
        tot_pol = tot_kl = 0.0
        steps = 0
        for i in range(0, len(tr_idx), args.batch):
            idx = tr_idx[i:i + args.batch]
            xb, mb, ab = batch(idx)
            A = torch.from_numpy(adv[idx]).to(dev)
            lp = logp(net, xb, mb)
            with torch.no_grad():
                lp0 = logp(net0, xb, mb)
            lpa = lp[torch.arange(len(ab), device=dev), ab]
            if args.method == "pg":
                pol = -(A * lpa).mean()
            elif args.method == "ppo":
                rho = torch.exp(lpa - lp0[torch.arange(len(ab), device=dev), ab])
                pol = -torch.min(rho * A, rho.clamp(1 - args.clip, 1 + args.clip) * A).mean()
            else:
                w = torch.exp(A / args.beta).clamp(max=20.0)
                pol = -(w * lpa).sum() / w.sum()
            kl = torch.where(mb, lp0.exp() * (lp0 - lp), 0.0).sum(1).mean()
            loss = pol + args.kl * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_pol += pol.item()
            tot_kl += kl.item()
            steps += 1
        vkl, vchg, dwin, dloss = evaluate()
        print(f"epoch {ep + 1}: policy loss {tot_pol / steps:.4f}, train KL {tot_kl / steps:.4f} | held-out games: "
              f"KL {vkl:.4f}, top move changed {vchg:.1%}, log p(move played) {dwin:+.3f} in wins, "
              f"{dloss:+.3f} in losses ({time.time() - t:.0f}s)", flush=True)

    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
              os.path.join(args.out, "weights", "model.safetensors"))
    json.dump(vars(args), open(os.path.join(args.out, "finetune_args.json"), "w"), indent=1)
    print(f"saved to {args.out}/weights/model.safetensors")


if __name__ == "__main__":
    main()
