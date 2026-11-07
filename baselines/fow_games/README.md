# Fog-of-War games dataset

Full games between a trained agent and the reference bots, with every turn of the agent kept in
order. Unlike `baselines/imitation`, which keeps random snapshots and labels them with a
full-board Stockfish teacher, this data records what the agent actually did in the fog and how
each game ended. It is meant for:

- **fine-tuning from game results** (PLAN.md, Idea 4): each row has the move played, its
  probability under the policy that played it, and the result of the game;
- **models that read a game turn by turn** (PLAN.md, Idea 3): a game's turns are consecutive
  rows, with the game id, ply and the agent's turn number.

## Scripts

| Script | What it does |
|---|---|
| `play_games.py` | plays the games and saves them (`games_*.jsonl.gz`), then encodes them (`shard_*.npz`) and writes `meta.json` |
| `encode_games.py` | re-encodes saved games for any agent's `features.py`, with no bots and no Stockfish |

```bash
export STOCKFISH=/path/to/stockfish
python baselines/fow_games/play_games.py --agent-dir agents/move_history_branch --games 60000 --workers 12 --seed 100 --out data/fow_games
# later, for a different input encoding:
python baselines/fow_games/encode_games.py --games data/fow_games --agent-dir agents/<new_agent> --out data/fow_games_<new_agent>
```

## Choices

- **Opponents:** bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000 in turn by game id; the
  agent alternates colours, so each opponent gets the same number of games with each colour.
- **Exploration:** the agent samples every move from `softmax(logits / 0.5)`. In 600-game tests
  per setting (agent's score out of 200 against greedy / guess-sf-4x500 / guess-sf-16x1000):
  temperature 0 (top move) 186 / 164 / 167, 0.3 186 / 173 / 147, 0.5 175 / 173 / 138,
  1.0 140 / 134 / 80. 0.5 tries other moves and loses more games while staying close to its
  real strength; 1.0 mostly records bad moves.
- **Seeds:** game seed = `seed * 1,000,000 + game id`, so a dataset never shares games with the
  arena evaluations (arena seeds 6 and 7). The Stockfish-based bots do not play the same moves on
  every run, so the saved moves, not the seeds, are the record.
- **Speed:** the agent's network runs inside each worker on one CPU thread (no separate process,
  no GPU), and no Stockfish labels are made unless `--teacher-nodes` is given.

## Shard columns

One row per agent turn, ordered by game and then turn. See the docstring of `encode_games.py`:
`x`, `move`, `legal`, `logp`, `game`, `ply`, `turn`, `plies_left`, `result` (1 / 0 / -1 for the
agent), `opponent`, `teacher` (-1 when not labelled) and `opponents`.

Checked on a pilot: recomputing the log-probability of every recorded move from the re-encoded
inputs matches the value recorded in play to within 4e-6.
