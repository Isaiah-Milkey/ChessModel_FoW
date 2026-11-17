"""Learn from Fog-of-War games in rounds, with no teacher: the turn-transformer agent plays fresh
games, learns from which moves led to wins, and plays again with the new weights.

    python baselines/fow_games/rl_loop.py --init agents/turn_transformer --out runs/rl_tf --rounds 10

Each round:
  1. Play --games new games (never replayed: every round has its own game seeds) against the
     reference bots and, for --league-share of them, against saved earlier versions of the model
     (the last --league snapshots, the starting model included). The learner samples every move
     from softmax(logits / --temperature), reading the game turn by turn exactly as
     agents/turn_transformer/agent.py does.
  2. Add them to the buffer and drop the oldest round, so the buffer holds the last --keep rounds
     (counted in games). Each game is trained on --epochs times per round, so at most
     --keep x --epochs times in total, and then never again.
  3. Value: a separate critic (a copy of the policy's CNN and transformer with its own head, plus
     the opponent) predicts the game result (1 win, 0 draw, -1 loss) at every agent turn. It is
     first scored on the new round's games (which it has not seen), then trained on the buffer.
  4. Advantages by generalized advantage estimation over each game's turns:
     delta_t = V(t+1) - V(t), with the game result after the last turn; A_t = sum_k lam^k delta_(t+k).
  5. Policy: clipped policy gradient (PPO): rho = p(move) / p_round(move), where p_round is the
     model that played the game, clipped to 1 +- --clip; plus --kl * KL(start || current) to keep
     it near the imitation model it started from.
  6. Save the round as an agent directory (rounds/rNNN); every --eval-every rounds rate it against
     the three bots with fow.arena on --eval-seed (not a training seed), and copy the best-rated
     round to best/.

Long games (draws run to 400 plies) would fill the buffer with their turns, so every turn is
weighted min(1, --turn-cap / turns in its game): a long game counts at most as much as a
--turn-cap-turn game. The value can be warm-started from games already played (--value-games,
e.g. data/fow_games), replayed and encoded with the agent's features.

Writes into --out: config.json, log.jsonl (one line per round), games/ (every game, reusable:
games_r<round>_<worker>.jsonl.gz in play_games.py's format), rounds/rNNN/ (agent directories),
best/, value.safetensors. The value is used only in training; it is not part of the agent.
"""
import argparse
import copy
import glob
import gzip
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines", "imitation"))
sys.path.insert(0, os.path.join(ROOT, "baselines", "sequence"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))
from generate_data import N_MOVES, load_encoding  # noqa: E402
from encode_seq import pack, unpack  # noqa: E402
from guard import Guard  # noqa: E402
from fow.bots import make_bot  # noqa: E402
from fow.game import Game  # noqa: E402

BOTS = ["bot:greedy", "bot:guess-sf-4x500", "bot:guess-sf-16x1000"]
SELF = len(BOTS)                      # opponent index of every saved version of our own model
GB_PER_WORKER = 0.7                   # memory of one game-playing worker (torch, the network, a bot)
ARENA_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # device for the rating games
GUARD = None                          # set in main(); pauses training while the PC is overloaded
SCORE = {"1-0": {"white": 1, "black": -1}, "0-1": {"white": -1, "black": 1},
         "1/2-1/2": {"white": 0, "black": 0}}
AGENT_FILES = ("agent.py", "features.py", "history.py", "model.py", "config.json")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_policy(agent_dir):
    m = load_module(os.path.join(agent_dir, "model.py"), "rl_model")
    net = m.PolicyNet(**json.load(open(os.path.join(agent_dir, "config.json"))))
    net.load_state_dict(load_file(os.path.join(agent_dir, "weights", "model.safetensors")))
    assert net.use_context, "the loop is written for the turn transformer (use_context true)"
    return net.eval()


# ---------------------------------------------------------------- playing (CPU worker processes)

class Learner:
    """The policy as the agent runs it (agent.py), but sampling every move at a temperature."""

    def __init__(self, agent_dir, temperature):
        self.net = load_policy(agent_dir)
        self.make_memory, self.enc, self.legal_idx, _ = load_encoding(agent_dir)
        self.t = temperature

    def reset(self, seed):
        self.mem, self.turns, self.rng = self.make_memory(), [], random.Random(seed)

    @torch.no_grad()
    def act(self, obs, force=None):
        """Sample a move; with `force` (a move of a recorded opening, for a mid-game start) play that
        move instead, still reading the turn so memory and turn context are built as in a real game."""
        self.mem.update(obs)
        x = self.enc(obs, self.mem)
        h = self.net.trunk(torch.from_numpy(x).float().unsqueeze(0))
        self.turns = (self.turns + [self.net.embed(h)])[-self.net.context:]
        c = self.net.turn_context(torch.cat(self.turns).unsqueeze(0))[0, -1:]
        logits = self.net.policy(h, c)[0]
        legal = self.legal_idx(obs)
        idx = sorted(legal)
        lp = torch.log_softmax(logits[idx] / self.t, 0)
        if force is None:
            k = self.rng.choices(range(len(idx)), weights=lp.exp().tolist())[0]
            mv = legal[idx[k]]
        else:                                             # under-promotions are scored as the queen promotion
            want = force if len(force) == 4 else force[:4] + "q"
            k = next(j for j, i in enumerate(idx) if legal[i] == want)
            mv = force
        mask = np.zeros(N_MOVES, np.uint8)
        mask[idx] = 1
        return mv, dict(bits=pack(x, obs["ply"]), ply=obs["ply"], move=idx[k],
                        legal=np.packbits(mask), logp_b=float(lp[k]), learn=force is None)


def play_worker(job):
    wid, games, cfg = job
    torch.set_num_threads(1)
    me_net = Learner(cfg["policy"], cfg["temperature"])
    opps = {}
    rows, g_off, g_res, g_opp, g_id = [], [0], [], [], []
    path = os.path.join(cfg["games_dir"], f"games_r{cfg['round']}_{wid}.jsonl.gz")
    with gzip.open(path, "wt") as f:
        for gid, gseed, opp, me, prefix in games:
            if opp not in opps:
                if opp.startswith("bot:"):
                    opps[opp] = make_bot(opp)
                else:                                     # a saved version of our model, as the agent plays
                    for name in ("features", "history", "model"):     # agent.py imports these by plain name;
                        sys.modules.pop(name, None)                  # don't reuse another agent's modules
                    sys.path.insert(0, opp)
                    opps[opp] = load_module(os.path.join(opp, "agent.py"), "rl_agent").Agent(opp, "cpu")
            bot = opps[opp]
            bot.reset("black" if me == "white" else "white", gseed ^ 0x5BD1E995)
            me_net.reset(gseed)
            game, moves, logps = Game(), [], []
            while not game.over:
                obs = game.observation()
                force = prefix[len(moves)] if len(moves) < len(prefix) else None   # mid-game start
                if obs["color"] == me:
                    mv, row = me_net.act(obs, force)
                    rows.append(row)
                    logps.append(round(row["logp_b"], 6))
                else:
                    mv = bot.act(obs)                     # the opponent also reads the forced opening
                    mv = force or mv
                moves.append(mv)
                game.play(mv)
            g_off.append(len(rows))
            g_res.append(SCORE[game.result][me])
            g_opp.append(BOTS.index(opp) if opp in BOTS else SELF)
            g_id.append(gid)
            f.write(json.dumps({"id": gid, "seed": gseed, "opponent": opp, "agent_color": me, "moves": moves,
                                "result": game.result, "termination": game.termination, "plies": len(moves),
                                "temperature": cfg["temperature"], "agent_logp": logps,
                                "start_ply": len(prefix)}) + "\n")
    for o in opps.values():
        if hasattr(o, "close"):
            o.close()
    return dict(bits=np.stack([r["bits"] for r in rows]), ply=np.array([r["ply"] for r in rows], np.int16),
                move=np.array([r["move"] for r in rows], np.int64), legal=np.stack([r["legal"] for r in rows]),
                logp_b=np.array([r["logp_b"] for r in rows], np.float32),
                learn=np.array([r["learn"] for r in rows], np.uint8), g_off=np.array(g_off, np.int64),
                g_res=np.array(g_res, np.float32), g_opp=np.array(g_opp, np.int64), g_id=np.array(g_id, np.int64))


def encode_record_file(job):
    """Agent-side turns of recorded games (play_games.py format): for warming up the value, and
    with full=True (moves, legal masks, logged probabilities) for rebuilding the buffer on --resume."""
    path, agent_dir, full = job
    make_memory, enc, legal_idx, move_idx = load_encoding(agent_dir)
    bits, ply, move, legal, logp_b, g_off, g_res, g_opp, g_id = [], [], [], [], [], [0], [], [], []
    learn = []
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            me, mem, game, turn = rec["agent_color"], make_memory(), Game(), 0
            start = rec.get("start_ply", 0)               # turns of a forced opening are context only
            for mv in rec["moves"]:
                obs = game.observation()
                if obs["color"] == me:
                    mem.update(obs)
                    bits.append(pack(enc(obs, mem), obs["ply"]))
                    ply.append(obs["ply"])
                    learn.append(obs["ply"] >= start)
                    if full:
                        mask = np.zeros(N_MOVES, np.uint8)
                        mask[list(legal_idx(obs))] = 1
                        legal.append(np.packbits(mask))
                        move.append(move_idx(mv if len(mv) == 4 else mv[:4] + "q", me))
                        logp_b.append(rec["agent_logp"][turn])
                    turn += 1
                game.play(mv)
            g_off.append(len(bits))
            g_res.append(SCORE[rec["result"]][me])
            g_opp.append(BOTS.index(rec["opponent"]) if rec["opponent"] in BOTS else SELF)
            g_id.append(rec["id"])
    out = dict(bits=np.stack(bits), ply=np.array(ply, np.int16), g_off=np.array(g_off, np.int64),
               g_res=np.array(g_res, np.float32), g_opp=np.array(g_opp, np.int64), g_id=np.array(g_id, np.int64))
    if full:
        out.update(move=np.array(move, np.int64), legal=np.stack(legal), logp_b=np.array(logp_b, np.float32),
                   learn=np.array(learn, np.uint8))
    return out


def load_openings(pattern, lo, hi, min_left):
    """Move lists of recorded games long enough to start a game from ply lo..hi with min_left plies to go."""
    out = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with gzip.open(path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                if rec["plies"] >= lo + min_left and rec.get("start_ply", 0) == 0:
                    out.append(rec["moves"])
    return out


def join(parts, keys):
    """Concatenate per-worker (or per-round) results, shifting the game offsets."""
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys if k != "g_off"}
    offs, base = [np.zeros(1, np.int64)], 0
    for p in parts:
        offs.append(p["g_off"][1:] + base)
        base += p["g_off"][-1]
    out["g_off"] = np.concatenate(offs)
    return out


# ---------------------------------------------------------------- networks on batches of games (GPU)

def chunks_of(g_off, context):
    """(first row, length) of every chunk of at most `context` turns, game by game, and the game of each."""
    out, game = [], []
    for g in range(len(g_off) - 1):
        for a in range(g_off[g], g_off[g + 1], context):
            out.append((a, min(a + context, g_off[g + 1]) - a))
            game.append(g)
    return np.array(out, np.int64), np.array(game, np.int64)


def features(body, x, lens):
    """CNN features h and transformer outputs c for the turns of chunks laid end to end."""
    h = body.trunk(x)
    e = body.embed(h)
    dev = x.device
    bi = torch.from_numpy(np.repeat(np.arange(len(lens)), lens)).to(dev)
    ti = torch.from_numpy(np.concatenate([np.arange(n) for n in lens])).to(dev)
    E = torch.zeros(len(lens), int(lens.max()), e.shape[1], device=dev, dtype=e.dtype)
    E[bi, ti] = e
    return h, body.turn_context(E)[bi, ti]


class Critic(nn.Module):
    """Value: a copy of the policy's CNN and transformer, with a head that also sees the opponent."""

    def __init__(self, policy, n_opp=SELF + 1):
        super().__init__()
        self.body = copy.deepcopy(policy)
        ch, d = policy.film.out_features // 2, policy.film.in_features
        self.h = nn.Sequential(nn.Conv2d(ch, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.Flatten(),
                               nn.Linear(32 * 64, 128))
        self.c = nn.Linear(d, 128)
        self.opp = nn.Embedding(n_opp, 128)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x, lens, opp):
        h, c = features(self.body, x, lens)
        return self.out(self.h(h) + self.c(c) + self.opp(opp))[:, 0]


def batches(n_chunks, size, rng):
    order = rng.permutation(n_chunks)
    return [order[i:i + size] for i in range(0, n_chunks, size)]


def gather(buf, ch, sel, dev):
    """Rows of the selected chunks: input planes, chunk lengths, row indices."""
    cs = ch[sel]
    rows = np.concatenate([np.arange(a, a + n) for a, n in cs])
    x = torch.from_numpy(unpack(buf["bits"][rows], buf["ply"][rows])).float().to(dev)
    return x, cs[:, 1], rows


def predict_values(critic, buf, ch, opp_row, dev, size=256):
    v = np.empty(len(buf["ply"]), np.float32)
    critic.eval()
    with torch.no_grad():
        for i in range(0, len(ch), size):
            x, lens, rows = gather(buf, ch, np.arange(i, min(i + size, len(ch))), dev)
            v[rows] = critic(x, lens, torch.from_numpy(opp_row[rows]).to(dev)).cpu().numpy()
    return v


def per_row(buf):
    """Game index, result, opponent and weight of every row."""
    g_len = np.diff(buf["g_off"])
    g_of = np.repeat(np.arange(len(g_len)), g_len)
    return g_of, buf["g_res"][g_of], buf["g_opp"][g_of], g_len[g_of]


def train_critic(critic, opt, buf, ch, w_row, dev, epochs, rng, size=96):
    _, res, opp, _ = per_row(buf)
    critic.train()
    for m in critic.body.modules():                       # the copied CNN keeps its batch-norm statistics
        if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
            m.eval()
    tot = wsum = 0.0
    for _ in range(epochs):
        for i, sel in enumerate(batches(len(ch), size, rng)):
            if i % 50 == 0:
                GUARD.wait_ok()
            x, lens, rows = gather(buf, ch, sel, dev)
            v = critic(x, lens, torch.from_numpy(opp[rows]).to(dev))
            w = torch.from_numpy(w_row[rows]).to(dev)
            loss = (w * (v - torch.from_numpy(res[rows]).to(dev)) ** 2).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * w.sum().item()
            wsum += w.sum().item()
    critic.eval()
    return tot / wsum


def gae(buf, v, lam):
    """A_t = sum_k lam^k delta_(t+k), delta_t = V(t+1) - V(t), the game result after the last turn."""
    adv = np.zeros(len(v), np.float32)
    for g in range(len(buf["g_off"]) - 1):
        a, b = buf["g_off"][g], buf["g_off"][g + 1]
        nxt = np.r_[v[a + 1:b], buf["g_res"][g]]
        delta = nxt - v[a:b]
        carry = 0.0
        for t in range(b - a - 1, -1, -1):
            carry = delta[t] + lam * carry
            adv[a + t] = carry
    return adv


def masked_logp(net, buf, x, lens, rows, dev):
    h, c = features(net, x, lens)
    mb = torch.from_numpy(np.unpackbits(buf["legal"][rows], axis=1)).bool().to(dev)
    return F.log_softmax(net.policy(h, c).masked_fill(~mb, -1e9), 1), mb


def round_logp(net, buf, ch, dev, size=256):
    """log p(move played) under `net` at temperature 1: the reference for the PPO ratio."""
    out = np.empty(len(buf["ply"]), np.float32)
    with torch.no_grad():
        for i in range(0, len(ch), size):
            x, lens, rows = gather(buf, ch, np.arange(i, min(i + size, len(ch))), dev)
            lp, _ = masked_logp(net, buf, x, lens, rows, dev)
            out[rows] = lp[torch.arange(len(rows), device=dev), torch.from_numpy(buf["move"][rows]).to(dev)].cpu().numpy()
    return out


# ---------------------------------------------------------------- the loop

def save_agent(net, init, path, note):
    os.makedirs(os.path.join(path, "weights"), exist_ok=True)
    for f in AGENT_FILES:
        shutil.copy(os.path.join(init, f), path)
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()},
              os.path.join(path, "weights", "model.safetensors"))
    open(os.path.join(path, "MODEL.md"), "w").write(f"# placeholder: {note}\n")


def arena_rating(agent_dir, games, seed, workers):
    """Rating against the three bots from fow.arena (and the agent's score against each)."""
    cmd = [sys.executable, "-m", "fow.arena", agent_dir, *BOTS, "--games", str(games), "--device", ARENA_DEVICE,
           "--workers", str(workers), "--seed", str(seed)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.replace("\\", "/")
    name = re.escape(agent_dir.replace("\\", "/"))
    m = re.search(rf"^\s+{name}\s+(-?\d+)\s+\[\s*(-?\d+),\s*(-?\d+)\]", out, re.M)
    scores = {b: pair_score(out, agent_dir, b) for b in BOTS}
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), scores) if m else (None, None, None, scores)


def pair_score(out, a, b):
    """a's score against b from a fow.arena results line ("X vs Y  s / n" gives X's score), or None."""
    a, b = a.replace("\\", "/"), b.replace("\\", "/")
    for x, y in ((a, b), (b, a)):
        m = re.search(rf"^\s+{re.escape(x)}\s+vs\s+{re.escape(y)}\s+([\d.]+) / (\d+)", out, re.M)
        if m:
            s = float(m.group(1)) / int(m.group(2))
            return round(s if x == a else 1 - s, 3)
    return None


def holdout_score(agent_dir, other, games, seed, workers):
    """Score against an agent the loop never plays (a check that it is not just learning the bots)."""
    cmd = [sys.executable, "-m", "fow.arena", agent_dir, other, "--games", str(games), "--device", ARENA_DEVICE,
           "--workers", str(workers), "--seed", str(seed)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.replace("\\", "/")
    return pair_score(out, agent_dir, other)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="agents/turn_transformer", help="starting agent (also the KL anchor)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--games", type=int, default=6000, help="new games per round")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.5, help="the learner samples from softmax(logits / T)")
    ap.add_argument("--league", type=int, default=3, help="saved versions of our model used as opponents")
    ap.add_argument("--league-share", type=float, default=0.25, help="share of games against those versions")
    ap.add_argument("--keep", type=int, default=2, help="rounds of games kept in the buffer")
    ap.add_argument("--epochs", type=int, default=2, help="policy passes over the buffer per round")
    ap.add_argument("--value-epochs", type=int, default=1, help="value passes over the buffer per round")
    ap.add_argument("--value-games", default="", help="recorded games (play_games.py) to warm up the value on")
    ap.add_argument("--value-warmup-epochs", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.8, help="GAE lambda")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl", type=float, default=0.1, help="weight of KL(start || current)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--value-lr", type=float, default=3e-4)
    ap.add_argument("--turn-cap", type=int, default=40, help="a game's turns count at most this much")
    ap.add_argument("--batch", type=int, default=96, help="chunks of games per batch")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-games", type=int, default=200, help="arena games against each bot")
    ap.add_argument("--eval-seed", type=int, default=9, help="arena seed for choosing the best round")
    ap.add_argument("--seed", type=int, default=7, help="game seeds are (seed * 1000 + round) * 1e6 + game number")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many ratings in a row without a new best (0: run all --rounds)")
    ap.add_argument("--holdout", default="", help="agent never played in training; scored against at each rating")
    ap.add_argument("--holdout-games", type=int, default=200)
    ap.add_argument("--value-init", default="", help="value weights to start from (value.safetensors of an earlier run)")
    ap.add_argument("--midgame-share", type=float, default=0.0,
                    help="share of games that start from a recorded game's position at ply --midgame-plies")
    ap.add_argument("--midgame-games", default="", help="glob of recorded games (games_*.jsonl.gz) to take openings from")
    ap.add_argument("--midgame-plies", default="30,60", help="start ply range, lo,hi")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in --out from its last finished round (its saved settings are used, "
                         "except --rounds, --workers, --patience and --holdout)")
    args = ap.parse_args()
    global GUARD
    GUARD = Guard("rl_loop")                              # one heavy job at a time; pauses while overloaded
    done = []                                             # log entries of the rounds already finished
    if args.resume:
        saved = json.load(open(os.path.join(args.out, "config.json")))
        defaults = vars(ap.parse_args(["--out", args.out]))      # settings added after the run started
        args = argparse.Namespace(**{**defaults, **saved, "out": args.out, "rounds": args.rounds, "workers": args.workers,
                                     "patience": args.patience, "holdout": args.holdout,
                                     "holdout_games": args.holdout_games, "resume": True})
        path = os.path.join(args.out, "log.jsonl")
        done = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
        nxt = len(done) + 1                               # remove what the interrupted round left behind
        shutil.rmtree(os.path.join(args.out, "rounds", f"r{nxt:03d}"), ignore_errors=True)
        for f in glob.glob(os.path.join(args.out, "games", f"games_r{nxt}_*.jsonl.gz")):
            os.remove(f)
        print(f"resuming after round {len(done)}", flush=True)
    else:
        if os.path.exists(args.out) and os.listdir(args.out):
            ap.error(f"{args.out} is not empty (use --resume to continue it)")
        for d in ("games", "rounds"):
            os.makedirs(os.path.join(args.out, d))
        json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = torch.device(args.device)
    torch.backends.cudnn.allow_tf32 = False               # full float32 on the GPU, as the agent computes on the CPU
    torch.backends.cuda.matmul.allow_tf32 = False

    net = load_policy(args.init).to(dev)
    anchor = load_policy(args.init).to(dev)
    for p in anchor.parameters():
        p.requires_grad_(False)
    critic = Critic(net).to(dev)
    v_opt = torch.optim.AdamW(critic.parameters(), lr=args.value_lr, weight_decay=1e-4)
    p_opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    league = [os.path.abspath(args.init)]
    keys = ("bits", "ply", "move", "legal", "logp_b", "logp_r", "learn", "g_res", "g_opp", "g_id", "g_off")
    value_path = os.path.join(args.out, "value.safetensors")
    openings, mid_lo, mid_hi = [], *map(int, args.midgame_plies.split(","))
    if args.midgame_share > 0:
        openings = load_openings(args.midgame_games, mid_lo, mid_hi, 10)
        print(f"mid-game starts: {len(openings):,} recorded games of at least {mid_lo + 10} plies", flush=True)
        assert openings, f"no recorded games match {args.midgame_games}"

    if os.path.exists(value_path):                        # resuming: the value as saved after the last round
        critic.load_state_dict(load_file(value_path))
    elif args.value_init:                                 # start from an earlier run's value
        critic.load_state_dict(load_file(args.value_init))
        save_file({k: v.detach().cpu().contiguous() for k, v in critic.state_dict().items()}, value_path)
    elif args.value_games:                                # warm up the value on games already played
        t = time.time()
        GUARD.wait_ok()
        files = sorted(glob.glob(os.path.join(args.value_games, "games_*.jsonl.gz")))
        with mp.get_context("spawn").Pool(min(GUARD.workers(args.workers, GB_PER_WORKER), len(files))) as pool:
            parts = pool.map(encode_record_file, [(f, args.init, False) for f in files])
        wb = join(parts, ("bits", "ply", "g_res", "g_opp", "g_off"))
        ch, _ = chunks_of(wb["g_off"], net.context)
        _, _, _, glen = per_row(wb)
        mse = train_critic(critic, v_opt, wb, ch, np.minimum(1, args.turn_cap / glen).astype(np.float32), dev,
                           args.value_warmup_epochs, rng)
        print(f"value warm-up on {len(wb['g_res']):,} games ({len(wb['ply']):,} turns) from {args.value_games}: "
              f"train MSE {mse:.4f} ({time.time() - t:.0f}s)", flush=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in critic.state_dict().items()}, value_path)
        del wb, parts

    buffer, best, since_best = [], (-1e9, None), 0       # since_best: ratings since the best one
    if done:                                              # resuming: weights, league, best round and buffer
        last = os.path.join(args.out, "rounds", f"r{len(done):03d}")
        net.load_state_dict(load_file(os.path.join(last, "weights", "model.safetensors")))
        league += [os.path.abspath(os.path.join(args.out, "rounds", f"r{k:03d}")) for k in range(1, len(done) + 1)]
        for e in done:
            if e.get("elo") is not None:
                since_best += 1
                if e["elo"] > best[0]:
                    best, since_best = (e["elo"], os.path.join(args.out, "rounds", f"r{e['round']:03d}")), 0
        for k in range(max(1, len(done) - args.keep + 1), len(done) + 1):
            GUARD.wait_ok()
            files = sorted(glob.glob(os.path.join(args.out, "games", f"games_r{k}_*.jsonl.gz")))
            with mp.get_context("spawn").Pool(min(GUARD.workers(args.workers, GB_PER_WORKER), len(files))) as pool:
                parts = pool.map(encode_record_file, [(f, args.init, True) for f in files])
            old = join(parts, [c for c in keys if c != "logp_r"])
            gen = load_policy(os.path.join(args.out, "rounds", f"r{k - 1:03d}") if k > 1 else args.init).to(dev)
            old["logp_r"] = round_logp(gen, old, chunks_of(old["g_off"], net.context)[0], dev)
            buffer.append(old)
            print(f"buffer: rebuilt round {k} ({len(old['g_res']):,} games)", flush=True)
            del gen

    for r in range(len(done) + 1, args.rounds + 1):
        t0 = time.time()
        GUARD.wait_ok()
        n_workers = GUARD.workers(args.workers, GB_PER_WORKER)
        cur = os.path.join(args.out, "rounds", f"r{r - 1:03d}") if r > 1 else args.init
        n_self = int(round(args.league_share * args.games))
        pairs = []                                        # (opponent, agent colour): each opponent gets both colours equally
        for pool_, n in ((BOTS, args.games - n_self), (league[-args.league:], n_self)):
            pairs += [(pool_[j % len(pool_)], "white" if (j // len(pool_)) % 2 == 0 else "black") for j in range(n)]
        base = (args.seed * 1000 + r) * 1_000_000
        mid = random.Random(base)                         # which games start mid-game, and from where
        games = []
        for i, (opp, me) in enumerate(pairs):
            prefix = []
            if openings and mid.random() < args.midgame_share:
                src = openings[mid.randrange(len(openings))]
                prefix = src[:mid.randint(mid_lo, min(mid_hi, len(src) - 10))]
            games.append((i, base + i, opp, me, prefix))
        cfg = {"policy": cur, "temperature": args.temperature, "round": r, "games_dir": os.path.join(args.out, "games")}
        jobs = [(w, games[w::n_workers], cfg) for w in range(n_workers)]
        with mp.get_context("spawn").Pool(n_workers) as pool:
            parts = pool.map(play_worker, jobs)
        new = join(parts, ("bits", "ply", "move", "legal", "logp_b", "learn", "g_res", "g_opp", "g_id"))
        t_play = time.time() - t0

        net.eval()
        ch_new, _ = chunks_of(new["g_off"], net.context)
        new["logp_r"] = round_logp(net, new, ch_new, dev)
        if r == 1:                                        # the batch computation must match what the learner played
            with torch.no_grad():
                sel = np.arange(min(64, len(ch_new)))
                x, lens, rows = gather(new, ch_new, sel, dev)
                h, c = features(net, x, lens)
                lg = net.policy(h, c)
                mb = torch.from_numpy(np.unpackbits(new["legal"][rows], axis=1)).bool().to(dev)
                lpT = F.log_softmax(lg.masked_fill(~mb, -1e9) / args.temperature, 1)
                lpT = lpT[torch.arange(len(rows), device=dev), torch.from_numpy(new["move"][rows]).to(dev)].cpu().numpy()
            gap = float(np.abs(lpT - new["logp_b"][rows]).max())
            print(f"check: played log-probabilities vs batch computation, max difference {gap:.2e}", flush=True)
            assert gap < 1e-3, "the training computation does not match the agent's play"

        g_of, res, opp, glen = per_row(new)
        v_new = predict_values(critic, new, ch_new, opp, dev)
        v_mse_new = float(np.mean((v_new - res) ** 2))   # on games the value has never seen
        score = {}
        for o in range(SELF + 1):
            m = new["g_opp"] == o
            if m.any():
                score[(BOTS + ["self"])[o]] = round(float((new["g_res"][m] + 1).mean() / 2), 3)
        draws = float(np.mean(new["g_res"] == 0))

        buffer = (buffer + [new])[-args.keep:]
        buf = join(buffer, keys)
        ch, _ = chunks_of(buf["g_off"], net.context)
        g_of, res, opp, glen = per_row(buf)
        w_row = np.minimum(1, args.turn_cap / glen).astype(np.float32)
        w_pol = w_row * buf["learn"]                      # the forced opening of a mid-game start is context only
        v_mse = train_critic(critic, v_opt, buf, ch, w_row, dev, args.value_epochs, rng)
        v = predict_values(critic, buf, ch, opp, dev)
        adv = gae(buf, v, args.lam)

        net.eval()                                        # dropout off, batch-norm statistics frozen
        stats = {"pol": 0.0, "kl": 0.0, "clipped": 0.0, "w": 0.0}
        for _ in range(args.epochs):
            for i, sel in enumerate(batches(len(ch), args.batch, rng)):
                if i % 50 == 0:
                    GUARD.wait_ok()
                x, lens, rows = gather(buf, ch, sel, dev)
                lp, mb = masked_logp(net, buf, x, lens, rows, dev)
                with torch.no_grad():
                    lp0, _ = masked_logp(anchor, buf, x, lens, rows, dev)
                ar = torch.arange(len(rows), device=dev)
                mv = torch.from_numpy(buf["move"][rows]).to(dev)
                rho = torch.exp(lp[ar, mv] - torch.from_numpy(buf["logp_r"][rows]).to(dev))
                A = torch.from_numpy(adv[rows]).to(dev)
                w = torch.from_numpy(w_pol[rows]).to(dev)
                if w.sum() == 0:
                    continue
                pol = -torch.min(rho * A, rho.clamp(1 - args.clip, 1 + args.clip) * A)
                kl = torch.where(mb, lp0.exp() * (lp0 - lp), 0.0).sum(1)
                loss = (w * (pol + args.kl * kl)).sum() / w.sum()
                p_opt.zero_grad()
                loss.backward()
                p_opt.step()
                ws = w.sum().item()
                stats["pol"] += (w * pol).sum().item()
                stats["kl"] += (w * kl).sum().item()
                stats["clipped"] += (w * ((rho - 1).abs() > args.clip).float()).sum().item()
                stats["w"] += ws

        rdir = os.path.join(args.out, "rounds", f"r{r:03d}")
        save_agent(net, args.init, rdir, f"rl_loop round {r} from {args.init}")
        league.append(os.path.abspath(rdir))
        entry = {"round": r, "games": len(new["g_res"]), "turns": len(new["ply"]), "buffer_games": len(buf["g_res"]),
                 "midgame_games": sum(1 for g in games if g[4]), "learned_turns": int(new["learn"].sum()),
                 "score_while_sampling": score, "draws": round(draws, 3),
                 "value_mse_new_games": round(v_mse_new, 4), "value_mse_train": round(v_mse, 4),
                 "mean_abs_adv": round(float(np.abs(adv).mean()), 4),
                 "policy_loss": round(stats["pol"] / stats["w"], 5), "kl_to_start": round(stats["kl"] / stats["w"], 4),
                 "clipped": round(stats["clipped"] / stats["w"], 3),
                 "minutes_play": round(t_play / 60, 1), "minutes_total": round((time.time() - t0) / 60, 1)}
        if r % args.eval_every == 0 or r == args.rounds:
            GUARD.wait_ok()
            elo, lo, hi, sc = arena_rating(os.path.relpath(rdir, ROOT), args.eval_games, args.eval_seed,
                                           GUARD.workers(6, GB_PER_WORKER))
            entry.update({"elo": elo, "elo_ci": [lo, hi], "eval_scores": sc})
            if args.holdout:
                GUARD.wait_ok()
                entry["holdout_score"] = holdout_score(os.path.relpath(rdir, ROOT), args.holdout, args.holdout_games,
                                                       args.eval_seed, GUARD.workers(6, GB_PER_WORKER))
            since_best += 1
            if elo is not None and elo > best[0]:
                best, since_best = (elo, rdir), 0
                shutil.rmtree(os.path.join(args.out, "best"), ignore_errors=True)
                shutil.copytree(rdir, os.path.join(args.out, "best"))
        save_file({k: v.detach().cpu().contiguous() for k, v in critic.state_dict().items()}, value_path)
        log.write(json.dumps(entry) + "\n")              # written last: a round counts as finished only now
        log.flush()
        print(json.dumps(entry), flush=True)
        if args.patience and since_best >= args.patience:
            print(f"plateau: {since_best} ratings in a row without a new best; stopping", flush=True)
            break
    print(f"best round: {best[1]} ({best[0]})")


if __name__ == "__main__":
    main()
