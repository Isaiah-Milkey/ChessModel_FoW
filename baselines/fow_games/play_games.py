"""Play Fog-of-War games between a trained agent and the reference bots, and save them as a
dataset that keeps every turn in order: for learning from game results (fine-tuning) and for
models that read a game turn by turn.

    python baselines/fow_games/play_games.py --agent-dir agents/move_history_branch --games 60000 --workers 12 --out data/fow_games

Each worker writes
  games_<seed>_<worker>.jsonl.gz   one line per game: id, seed, opponent, agent colour, every move,
                                   result, termination, and for each agent turn the log-probability
                                   of the move it played (and Stockfish's move, with --teacher-nodes).
                                   This is the reusable record: encode_games.py rebuilds training
                                   arrays from it for any agent's encoding without playing again.
  shard_<seed>_<worker>.npz        those games encoded with --agent-dir's features (encode_games.py)
and the run writes meta.json (settings, commit, weight checksum, scores).

The agent's network runs inside the worker (no separate process) on one CPU thread. It samples
every move from softmax(logits / temperature) with a generator seeded by the game seed, so it
tries alternatives and every game can be replayed; --temperature 0 always plays the top move.
Opponents take turns by game id, and the agent alternates colours, so every opponent gets the
same number of games with each colour.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines", "imitation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_data import load_encoding, teacher_move  # noqa: E402
from encode_games import SCORE, encode_file  # noqa: E402
from fow.bots import make_bot, stockfish_path  # noqa: E402
from fow.game import Game  # noqa: E402

DEFAULT_OPPONENTS = "bot:greedy,bot:guess-sf-4x500,bot:guess-sf-16x1000"


def load_net(agent_dir):
    spec = importlib.util.spec_from_file_location("agent_model", os.path.join(agent_dir, "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = json.load(open(os.path.join(agent_dir, "config.json")))
    net = m.PolicyNet(cfg["in_planes"], cfg["channels"], cfg["blocks"])
    net.load_state_dict(load_file(os.path.join(agent_dir, "weights", "model.safetensors")))
    return net.eval()


def game_setup(gid, seed, opponents):
    """(game seed, opponent, agent colour) for game id gid."""
    k = len(opponents)
    return seed * 1_000_000 + gid, opponents[gid % k], "white" if (gid // k) % 2 == 0 else "black"


def worker(args):
    wid, ids, a = args
    torch.set_num_threads(1)
    net = load_net(a["agent_dir"])
    make_memory, enc, legal_idx, _ = load_encoding(a["agent_dir"])
    engine = None
    if a["teacher_nodes"]:
        import chess.engine
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
        engine.configure({"Threads": 1, "Hash": 16})
    bots = {}
    path = os.path.join(a["out"], f"games_{a['seed']}_{wid}.jsonl.gz")
    t0 = time.time()
    with gzip.open(path, "wt") as f:
        for n, gid in enumerate(ids):
            gseed, opp, me = game_setup(gid, a["seed"], a["opponents"])
            if opp not in bots:
                bots[opp] = make_bot(opp)
            bot = bots[opp]
            bot.reset("black" if me == "white" else "white", gseed ^ 0x5BD1E995)
            rng = random.Random(gseed)
            mem = make_memory()
            game = Game()
            moves, logps, labels = [], [], []
            while not game.over:
                obs = game.observation()
                if obs["color"] == me:
                    if engine is not None:
                        labels.append(teacher_move(engine, game, a["teacher_nodes"]))
                    mem.update(obs)
                    legal = legal_idx(obs)
                    idx = sorted(legal)
                    with torch.no_grad():
                        logits = net(torch.from_numpy(enc(obs, mem)).float().unsqueeze(0))[0]
                    if a["temperature"] > 0:
                        lp = torch.log_softmax(logits[idx] / a["temperature"], 0)
                        k = rng.choices(range(len(idx)), weights=lp.exp().tolist())[0]
                        logps.append(round(float(lp[k]), 6))
                    else:
                        k = int(torch.argmax(logits[idx]))
                        logps.append(0.0)
                    mv = legal[idx[k]]
                else:
                    mv = bot.act(obs)
                moves.append(mv)
                game.play(mv)
            rec = {"id": gid, "seed": gseed, "opponent": opp, "agent_color": me, "moves": moves,
                   "result": game.result, "termination": game.termination, "plies": len(moves),
                   "temperature": a["temperature"], "agent_logp": logps}
            if engine is not None:
                rec["teacher"] = labels
            f.write(json.dumps(rec) + "\n")
            if (n + 1) % 200 == 0:
                print(f"worker {wid}: {n + 1}/{len(ids)} games ({time.time() - t0:.0f}s)", flush=True)
    if engine is not None:
        engine.quit()
    for bot in bots.values():
        bot.close()
    n_games, n_rows = encode_file(path, a["out"], a["agent_dir"], a["opponents"])
    return n_games, n_rows


def summarize(out, opponents):
    """Agent's score per opponent and colour, from the game records."""
    score = {o: {"white": [0.0, 0], "black": [0.0, 0]} for o in opponents}
    plies = []
    for name in sorted(os.listdir(out)):
        if name.startswith("games_") and name.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(out, name), "rt") as f:
                for line in f:
                    r = json.loads(line)
                    s = score[r["opponent"]][r["agent_color"]]
                    s[0] += (SCORE[r["result"]][r["agent_color"]] + 1) / 2
                    s[1] += 1
                    plies.append(r["plies"])
    plies.sort()
    return score, plies[len(plies) // 2] if plies else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", default="agents/move_history_branch")
    ap.add_argument("--opponents", default=DEFAULT_OPPONENTS, help="comma-separated bot names")
    ap.add_argument("--games", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100,
                    help="dataset seed; game seeds are seed*1e6 + game id (arena evaluations use other seeds)")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="sample every agent move from softmax(logits / T); 0 plays the top move")
    ap.add_argument("--teacher-nodes", type=int, default=0,
                    help="also label every agent turn with full-board Stockfish at this many nodes (slow; 0 = off)")
    ap.add_argument("--out", default="data/fow_games")
    args = ap.parse_args()
    if os.path.exists(args.out) and os.listdir(args.out):
        ap.error(f"{args.out} is not empty")
    os.makedirs(args.out)
    opponents = args.opponents.split(",")
    a = {"agent_dir": args.agent_dir, "opponents": opponents, "seed": args.seed,
         "temperature": args.temperature, "teacher_nodes": args.teacher_nodes, "out": args.out}
    ids = list(range(args.games))
    jobs = [(w, ids[w::args.workers], a) for w in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        done = pool.map(worker, jobs)
    n_games, n_rows = sum(g for g, _ in done), sum(r for _, r in done)
    score, median_plies = summarize(args.out, opponents)
    weights = os.path.join(args.agent_dir, "weights", "model.safetensors")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip()
    except OSError:
        commit = None
    meta = {"args": vars(args), "opponents": opponents, "games": n_games, "agent_turns": n_rows,
            "median_plies": median_plies, "minutes": round((time.time() - t0) / 60, 1), "commit": commit,
            "weights_sha256": hashlib.sha256(open(weights, "rb").read()).hexdigest(),
            "agent_score": {o: {c: f"{s[0]:g}/{s[1]}" for c, s in d.items()} for o, d in score.items()}}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
