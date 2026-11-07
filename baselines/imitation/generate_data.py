"""Generate imitation data: positions from games between reference bots, each labelled with the
move Stockfish chooses when it sees the WHOLE board (a privileged teacher: it knows things your
agent cannot see, so its labels are a useful but limited signal).

    python baselines/imitation/generate_data.py --games 2000 --workers 8 --out data/imitation

Inputs are encoded with agents/baseline/features.py and fow.memory.Memory. With --agent-dir,
they use that agent's features.py instead, and its make_memory() and encode_from_memory(obs,
memory) when it defines them, so the training inputs match what that agent sees in play.

With --record, each worker also writes games_<seed>_<worker>.jsonl.gz: one line per game with
its moves and the teacher label of every kept position. reencode.py turns those into shards
for any agent's encoding without running Stockfish again.

Options for more varied games (all off by default):
  --random-plies N     each game opens with 0..N random legal moves (the number is random per game)
  --keep-schedule S    keep probability by ply, e.g. "0:0.1,10:0.5,20:1" (replaces --keep)
  --matchups long      fewer quick wins: no random or greedy bots, more guess-sf games, and
                       --play-agent DIR (for example agents/move_history) plays in 3 of 4 pairings

Writes one .npz shard per worker with
    x      uint8 [N, 22, 8, 8]   input planes (see agents/baseline/features.py)
    y      int16 [N]             teacher move index
    legal  uint8 [N, 512]        packed bit mask of legal move indices
"""
import argparse
import gzip
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys

import chess
import chess.engine
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "agents", "baseline"))
from features import N_MOVES, encode, legal_indices, move_index  # noqa: E402
from fow.bots import make_bot, stockfish_path  # noqa: E402
from fow.game import Game  # noqa: E402
from fow.memory import Memory  # noqa: E402

# who plays the games the positions come from (pairs are sampled uniformly)
MATCHUPS = [("bot:guess-sf-4x500", "bot:guess-sf-4x500"), ("bot:guess-sf-4x500", "bot:greedy"),
            ("bot:naive-sf", "bot:guess-sf-4x500"), ("bot:greedy", "bot:greedy"),
            ("bot:guess-sf-4x500", "bot:random")]
# --matchups long: "agent" is the --play-agent directory
LONG_MATCHUPS = [("bot:guess-sf-4x500", "bot:guess-sf-4x500"), ("agent", "bot:guess-sf-4x500"),
                 ("agent", "bot:guess-sf-16x1000"), ("agent", "agent")]


class AgentPlayer:
    """An agent directory as a data-generation player, run in its own process as in fow.arena."""

    def __init__(self, agent_dir):
        from fow.referee import make_player
        self.player = make_player(agent_dir, device="cpu")

    def reset(self, color, seed):
        self.player.reset(color, seed)

    def act(self, obs):
        return self.player.act(obs, 120)[0]

    def close(self):
        self.player.close()


def parse_schedule(text):
    """"0:0.1,10:0.5,20:1" -> keep probability function of the ply."""
    steps = sorted((int(a), float(b)) for a, b in (part.split(":") for part in text.split(",")))
    return lambda ply: [p for start, p in steps if ply >= start][-1]


def load_encoding(agent_dir):
    """(make_memory, encode(obs, memory), legal_indices, move_index) for an agent directory;
    agent_dir None means the baseline's encoding."""
    if agent_dir is None:
        return Memory, lambda obs, mem: encode(obs, mem.last_seen), legal_indices, move_index
    sys.path.insert(0, os.path.abspath(agent_dir))   # lets features.py import its own modules
    spec = importlib.util.spec_from_file_location("agent_features", os.path.join(agent_dir, "features.py"))
    feats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feats)
    make_memory = getattr(feats, "make_memory", Memory)
    enc = getattr(feats, "encode_from_memory", lambda obs, mem: feats.encode(obs, mem.last_seen))
    return make_memory, enc, feats.legal_indices, feats.move_index


def teacher_move(engine, game, nodes):
    for mv in game.legal_uci():
        if game.board.piece_type_at(chess.parse_square(mv[2:4])) == chess.KING:
            return mv
    try:
        mv = engine.play(game.board.copy(stack=False), chess.engine.Limit(nodes=nodes)).move.uci()
    except Exception:
        return None
    return mv if mv in game.legal_uci() else None


def worker(args):
    wid, n_games, nodes, keep, out, seed, agent_dir, record, random_plies, schedule, matchups, play_agent = args
    keep_at = parse_schedule(schedule) if schedule else (lambda ply: keep)
    make_memory, enc, legal_idx, move_idx = load_encoding(agent_dir)
    rng = random.Random(seed + wid)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
    engine.configure({"Threads": 1, "Hash": 16})
    bots = {}
    xs, ys, legals = [], [], []
    games = []
    for g in range(n_games):
        a, b = rng.choice(LONG_MATCHUPS if matchups == "long" else MATCHUPS)
        if rng.random() < 0.5:
            a, b = b, a
        players = {}
        for color, spec in (("white", a), ("black", b)):
            key = (spec, color) if spec == "agent" else spec   # an agent needs one process per side
            if key not in bots:
                bots[key] = AgentPlayer(play_agent) if spec == "agent" else make_bot(spec)
            players[color] = bots[key]
        players["white"].reset("white", rng.randrange(1 << 30))
        players["black"].reset("black", rng.randrange(1 << 30))
        mem = {"white": make_memory(), "black": make_memory()}
        game = Game()
        moves, labels = [], []
        n_random = rng.randint(0, random_plies) if random_plies else 0
        while not game.over:
            obs = game.observation()
            c = obs["color"]
            mem[c].update(obs)
            if rng.random() < keep_at(obs["ply"]):
                label = teacher_move(engine, game, nodes)
                if label is not None:
                    if len(label) == 5:
                        label = label[:4] + "q"
                    mask = np.zeros(N_MOVES, np.uint8)
                    mask[list(legal_idx(obs))] = 1
                    xs.append(enc(obs, mem[c]))
                    ys.append(move_idx(label, c))
                    legals.append(np.packbits(mask))
                    labels.append([obs["ply"], label])
            mv = players[c].act(obs)               # always called, so players keep up with the game
            if obs["ply"] < n_random:
                mv = rng.choice(sorted(game.legal_uci()))
            moves.append(mv)
            game.play(mv)
        if record:
            rec = {"white": a, "black": b, "moves": moves, "labels": labels}
            if random_plies:
                rec["random_plies"] = n_random
            games.append(rec)
        if (g + 1) % 50 == 0:
            print(f"worker {wid}: {g + 1}/{n_games} games, {len(ys)} positions", flush=True)
    engine.quit()
    for bot in bots.values():
        bot.close()
    path = os.path.join(out, f"shard_{seed}_{wid}.npz")
    np.savez_compressed(path, x=np.stack(xs), y=np.array(ys, np.int16), legal=np.stack(legals))
    if record:
        with gzip.open(os.path.join(out, f"games_{seed}_{wid}.jsonl.gz"), "wt") as f:
            for rec in games:
                f.write(json.dumps(rec) + "\n")
    return len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=5000, help="Stockfish nodes per label")
    ap.add_argument("--keep", type=float, default=0.5, help="fraction of positions kept")
    ap.add_argument("--out", default="data/imitation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-dir", default=None,
                    help="encode inputs with this agent's features.py (default: agents/baseline)")
    ap.add_argument("--record", action="store_true",
                    help="also save every game's moves and teacher labels, for reencode.py")
    ap.add_argument("--random-plies", type=int, default=0,
                    help="open each game with 0..N random legal moves (N drawn per game)")
    ap.add_argument("--keep-schedule", default=None,
                    help='keep probability by ply, e.g. "0:0.1,10:0.5,20:1" (overrides --keep)')
    ap.add_argument("--matchups", choices=["default", "long"], default="default")
    ap.add_argument("--play-agent", default=None, help="agent directory that plays in --matchups long")
    args = ap.parse_args()
    if args.matchups == "long" and not args.play_agent:
        ap.error("--matchups long needs --play-agent")
    os.makedirs(args.out, exist_ok=True)
    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
    jobs = [(i, per[i], args.nodes, args.keep, args.out, args.seed, args.agent_dir, args.record,
             args.random_plies, args.keep_schedule, args.matchups, args.play_agent) for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        n = sum(pool.map(worker, jobs))
    print(f"wrote {n} positions to {args.out}")


if __name__ == "__main__":
    main()
