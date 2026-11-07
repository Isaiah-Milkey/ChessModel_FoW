"""Rebuild imitation shards from recorded games for any agent's encoding, without Stockfish.

generate_data.py --record saves each game's moves and the teacher label of every kept position.
This replays those moves, encodes the labelled positions with the agent's features.py (as
generate_data.py --agent-dir does) and attaches the recorded labels, so it writes the same
shards generate_data.py would have written with that encoding, in minutes.

    python baselines/imitation/reencode.py --games data/rec --agent-dir agents/move_history --out data/rec_hist

Without --agent-dir it uses the baseline encoding. games_<seed>_<worker>.jsonl.gz becomes
shard_<seed>_<worker>.npz.
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_data import N_MOVES, load_encoding  # noqa: E402
from fow.game import Game  # noqa: E402


def reencode(args):
    path, out, agent_dir = args
    make_memory, enc, legal_idx, move_idx = load_encoding(agent_dir)
    xs, ys, legals = [], [], []
    n_games = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            labels = {ply: label for ply, label in rec["labels"]}
            mem = {"white": make_memory(), "black": make_memory()}
            game = Game()
            for mv in rec["moves"]:
                obs = game.observation()
                c = obs["color"]
                mem[c].update(obs)
                label = labels.get(obs["ply"])
                if label is not None:
                    mask = np.zeros(N_MOVES, np.uint8)
                    mask[list(legal_idx(obs))] = 1
                    xs.append(enc(obs, mem[c]))
                    ys.append(move_idx(label, c))
                    legals.append(np.packbits(mask))
                game.play(mv)
            assert game.over, f"{path}: game {n_games} did not end after its recorded moves"
            n_games += 1
    name = os.path.basename(path).replace("games_", "shard_").replace(".jsonl.gz", ".npz")
    np.savez_compressed(os.path.join(out, name), x=np.stack(xs), y=np.array(ys, np.int16),
                        legal=np.stack(legals))
    print(f"{name}: {n_games} games, {len(ys)} positions", flush=True)
    return len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="directory with games_*.jsonl.gz from generate_data.py --record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent-dir", default=None,
                    help="encode inputs with this agent's features.py (default: agents/baseline)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.games, "games_*.jsonl.gz")))
    assert files, f"no games_*.jsonl.gz in {args.games}"
    os.makedirs(args.out, exist_ok=True)
    jobs = [(f, args.out, args.agent_dir) for f in files]
    with mp.get_context("spawn").Pool(min(args.workers, len(files))) as pool:
        n = sum(pool.map(reencode, jobs))
    print(f"wrote {n} positions to {args.out}")


if __name__ == "__main__":
    main()
