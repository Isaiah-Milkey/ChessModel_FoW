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
    wid, n_games, nodes, keep, out, seed, agent_dir, record = args
    make_memory, enc, legal_idx, move_idx = load_encoding(agent_dir)
    rng = random.Random(seed + wid)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
    engine.configure({"Threads": 1, "Hash": 16})
    bots = {}
    xs, ys, legals = [], [], []
    games = []
    for g in range(n_games):
        a, b = rng.choice(MATCHUPS)
        if rng.random() < 0.5:
            a, b = b, a
        for spec in (a, b):
            if spec not in bots:
                bots[spec] = make_bot(spec)
        players = {"white": bots[a], "black": bots[b]}
        players["white"].reset("white", rng.randrange(1 << 30))
        players["black"].reset("black", rng.randrange(1 << 30))
        mem = {"white": make_memory(), "black": make_memory()}
        game = Game()
        moves, labels = [], []
        while not game.over:
            obs = game.observation()
            c = obs["color"]
            mem[c].update(obs)
            if rng.random() < keep:
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
            mv = players[c].act(obs)
            moves.append(mv)
            game.play(mv)
        if record:
            games.append({"white": a, "black": b, "moves": moves, "labels": labels})
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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
    jobs = [(i, per[i], args.nodes, args.keep, args.out, args.seed, args.agent_dir, args.record) for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        n = sum(pool.map(worker, jobs))
    print(f"wrote {n} positions to {args.out}")


if __name__ == "__main__":
    main()
