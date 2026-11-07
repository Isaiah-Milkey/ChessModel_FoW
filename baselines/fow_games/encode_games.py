"""Turn recorded Fog-of-War games (from play_games.py) into training shards, keeping every turn
of the agent in order, for any agent's encoding.

    python baselines/fow_games/encode_games.py --games data/fow_games --agent-dir agents/move_history_branch --out data/fow_games_enc

Replays each game's moves with fow.game, encodes every observation of the agent's side with the
agent's features.py (as the agent sees it in play), and writes shard_<seed>_<worker>.npz per
games_<seed>_<worker>.jsonl.gz. No bots and no Stockfish are run, so this takes seconds per
shard and can be redone for a new encoding at any time.

Each shard holds one row per agent turn, ordered by game and then by turn, so a game's turns are
consecutive rows (a sequence):
    x          uint8   [N, planes, 8, 8]  input planes, exactly as the agent encodes them in play
    move       int16   [N]   index of the move the agent played
    legal      uint8   [N, 512]  packed bit mask of legal move indices
    logp       float32 [N]   log-probability of that move under the policy that played it
                             (at the recorded temperature; 0 when it took the top move)
    game       int32   [N]   game id (unique across the dataset)
    ply        int16   [N]   ply of the position (0 = White's first move)
    turn       int16   [N]   the agent's own turn number in this game (0, 1, 2, ...)
    plies_left int16   [N]   plies from this position to the end of the game
    result     int8    [N]   game result for the agent: 1 win, 0 draw, -1 loss
    opponent   int8    [N]   index into `opponents`
    teacher    int16   [N]   Stockfish's move index for this position, -1 if not labelled
    opponents  str     [K]   opponent names
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

SCORE = {"1-0": {"white": 1, "black": -1}, "0-1": {"white": -1, "black": 1},
         "1/2-1/2": {"white": 0, "black": 0}}


def encode_file(path, out, agent_dir, opponents):
    """Encode one games_*.jsonl.gz into one shard; returns (games, rows)."""
    make_memory, enc, legal_idx, move_idx = load_encoding(agent_dir)
    opp_index = {o: i for i, o in enumerate(opponents)}
    cols = {k: [] for k in ("x", "move", "legal", "logp", "game", "ply", "turn", "plies_left",
                            "result", "opponent", "teacher")}
    n_games = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            me = rec["agent_color"]
            mem = make_memory()
            game = Game()
            turn = 0
            for mv in rec["moves"]:
                obs = game.observation()
                if obs["color"] == me:
                    mem.update(obs)
                    legal = legal_idx(obs)
                    mask = np.zeros(N_MOVES, np.uint8)
                    mask[list(legal)] = 1
                    label = rec["teacher"][turn] if rec.get("teacher") else None
                    if label is not None and len(label) == 5:
                        label = label[:4] + "q"
                    cols["x"].append(enc(obs, mem))
                    cols["move"].append(move_idx(mv if len(mv) == 4 else mv[:4] + "q", me))
                    cols["legal"].append(np.packbits(mask))
                    cols["logp"].append(rec["agent_logp"][turn])
                    cols["game"].append(rec["id"])
                    cols["ply"].append(obs["ply"])
                    cols["turn"].append(turn)
                    cols["plies_left"].append(rec["plies"] - obs["ply"])
                    cols["result"].append(SCORE[rec["result"]][me])
                    cols["opponent"].append(opp_index[rec["opponent"]])
                    cols["teacher"].append(move_idx(label, me) if label else -1)
                    turn += 1
                game.play(mv)
            assert game.over and game.result == rec["result"], f"{path}: game {rec['id']} does not replay"
            assert turn == len(rec["agent_logp"]), f"{path}: game {rec['id']} has the wrong number of agent turns"
            n_games += 1
    types = {"move": np.int16, "logp": np.float32, "game": np.int32, "ply": np.int16, "turn": np.int16,
             "plies_left": np.int16, "result": np.int8, "opponent": np.int8, "teacher": np.int16}
    arrays = {k: np.array(v, types[k]) for k, v in cols.items() if k in types}
    arrays["x"] = np.stack(cols["x"]).astype(np.uint8)
    arrays["legal"] = np.stack(cols["legal"])
    arrays["opponents"] = np.array(opponents)
    name = os.path.basename(path).replace("games_", "shard_").replace(".jsonl.gz", ".npz")
    np.savez_compressed(os.path.join(out, name), **arrays)
    return n_games, len(arrays["move"])


def _job(args):
    return encode_file(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="directory with games_*.jsonl.gz and meta.json from play_games.py")
    ap.add_argument("--agent-dir", required=True, help="encode inputs with this agent's features.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.games, "games_*.jsonl.gz")))
    assert files, f"no games_*.jsonl.gz in {args.games}"
    opponents = json.load(open(os.path.join(args.games, "meta.json")))["opponents"]
    os.makedirs(args.out, exist_ok=True)
    with mp.get_context("spawn").Pool(min(args.workers, len(files))) as pool:
        done = pool.map(_job, [(f, args.out, args.agent_dir, opponents) for f in files])
    print(f"wrote {sum(r for _, r in done):,} turns from {sum(g for g, _ in done):,} games to {args.out}")


if __name__ == "__main__":
    main()
