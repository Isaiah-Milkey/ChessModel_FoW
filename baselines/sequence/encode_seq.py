"""Encode recorded games as turn sequences, one per side, for models that read a game turn by turn.

    python baselines/sequence/encode_seq.py --games data/branch54k_s4 --agent-dir agents/move_history_branch --out data/branch54k_s4_seq

Reads games_*.jsonl.gz written by baselines/imitation/generate_data.py --record (each game's moves
and the Stockfish label of every kept position). Each game is replayed and every turn of each side
is encoded with the agent's features.py, exactly as that side's agent would see it in play, so
each game gives two sequences (White's turns, Black's turns). Only the kept turns have a teacher
label; the others are context.

The encoding must be move_history's (29 planes: plane 22 holds min(ply, 100), all other planes are
0/1). To keep the data small, the 28 binary planes are bit-packed and plane 22 is rebuilt from the
ply when loading (see unpack()). One seq_<seed>_<worker>.npz per games file:
    bits     uint8 [N, 224]  the 28 binary planes, bit-packed (N = all turns of all sequences)
    ply      int16 [N]       ply of each turn
    label    int16 [N]       Stockfish's move index, -1 for turns without a label
    legal    uint8 [M, 512]  packed legal-move masks of the labelled turns only, in row order
    seq_off  int64 [S + 1]   sequence s is rows seq_off[s] .. seq_off[s + 1] - 1, in turn order
    seq_game int32 [S]       game number within this file
    seq_color int8 [S]       0 White, 1 Black
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines", "imitation"))
from generate_data import N_MOVES, load_encoding  # noqa: E402
from fow.game import Game  # noqa: E402

N_PLANES = 29
TURN_PLANE = 22
PLY_CAP = 100
BINARY = [p for p in range(N_PLANES) if p != TURN_PLANE]


def pack(x, ply):
    """[29, 8, 8] uint8 -> 224 packed bytes (checks that the encoding is move_history's)."""
    assert x.shape == (N_PLANES, 8, 8) and (x[TURN_PLANE] == min(ply, PLY_CAP)).all() and x[BINARY].max() <= 1
    return np.packbits(x[BINARY].reshape(-1))


def unpack(bits, ply):
    """Inverse of pack for many rows: [N, 224], [N] -> [N, 29, 8, 8] uint8."""
    n = len(bits)
    b = np.unpackbits(bits, axis=1).reshape(n, N_PLANES - 1, 64)
    x = np.empty((n, N_PLANES, 64), np.uint8)
    x[:, BINARY] = b
    x[:, TURN_PLANE] = np.minimum(ply, PLY_CAP).astype(np.uint8)[:, None]
    return x.reshape(n, N_PLANES, 8, 8)


def encode_file(args):
    path, out, agent_dir = args
    make_memory, enc, legal_idx, move_idx = load_encoding(agent_dir)
    bits, plies, labels, legals, off, games, colors = [], [], [], [], [0], [], []
    with gzip.open(path, "rt") as f:
        for g, line in enumerate(f):
            rec = json.loads(line)
            lab = {ply: mv for ply, mv in rec["labels"]}
            mem = {"white": make_memory(), "black": make_memory()}
            rows = {"white": [], "black": []}
            game = Game()
            for mv in rec["moves"]:
                obs = game.observation()
                c = obs["color"]
                mem[c].update(obs)
                label = lab.get(obs["ply"])
                if label is not None:
                    if len(label) == 5:
                        label = label[:4] + "q"
                    mask = np.zeros(N_MOVES, np.uint8)
                    mask[list(legal_idx(obs))] = 1
                    label = (move_idx(label, c), np.packbits(mask))
                rows[c].append((pack(enc(obs, mem[c]), obs["ply"]), obs["ply"], label))
                game.play(mv)
            assert game.over, f"{path}: game {g} did not end after its recorded moves"
            for ci, c in enumerate(("white", "black")):
                for b, ply, label in rows[c]:
                    bits.append(b)
                    plies.append(ply)
                    labels.append(label[0] if label else -1)
                    if label:
                        legals.append(label[1])
                off.append(len(bits))
                games.append(g)
                colors.append(ci)
    name = os.path.basename(path).replace("games_", "seq_").replace(".jsonl.gz", ".npz")
    np.savez_compressed(os.path.join(out, name), bits=np.stack(bits), ply=np.array(plies, np.int16),
                        label=np.array(labels, np.int16), legal=np.stack(legals),
                        seq_off=np.array(off, np.int64), seq_game=np.array(games, np.int32),
                        seq_color=np.array(colors, np.int8))
    n_lab = sum(1 for v in labels if v >= 0)
    print(f"{name}: {len(games)} sequences, {len(bits)} turns, {n_lab} labelled", flush=True)
    return len(games), len(bits), n_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="directory with games_*.jsonl.gz from generate_data.py --record")
    ap.add_argument("--agent-dir", default="agents/move_history_branch", help="agent whose features.py encodes the turns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.games, "games_*.jsonl.gz")))
    assert files, f"no games_*.jsonl.gz in {args.games}"
    os.makedirs(args.out, exist_ok=True)
    with mp.get_context("spawn").Pool(min(args.workers, len(files))) as pool:
        done = pool.map(encode_file, [(f, args.out, args.agent_dir) for f in files])
    print(f"wrote {sum(d[0] for d in done):,} sequences, {sum(d[1] for d in done):,} turns, "
          f"{sum(d[2] for d in done):,} labelled, to {args.out}")


if __name__ == "__main__":
    main()
