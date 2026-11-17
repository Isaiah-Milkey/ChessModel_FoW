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
| `value.py` | trains a value head (predicts the game result from a position and the opponent) on the policy's frozen trunk; used only in training |
| `finetune.py` | fine-tunes a policy on the games from their results (clipped policy gradient with a KL penalty to the starting policy), optionally judging moves with the value head (`--value`, `--lam`) |
| `rl_loop.py` | the same idea in rounds for `agents/turn_transformer`: play fresh games, train a value and the policy on the last few rounds, play again with the new weights (below) |

Best result so far (see JOURNAL.md, 2026-11-07): value head, then
`finetune.py --method ppo --gamma 1 --kl 0.1 --value runs/value_opp --lam 0.8 --epochs 2`
beat the starting `move_history_branch` 54.5% over 3,000 games (+31 Elo).

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

## Learning in rounds (`rl_loop.py`)

No teacher: the imitation model is only the starting point.

```bash
python baselines/fow_games/rl_loop.py --init agents/turn_transformer --out runs/rl_tf --rounds 40 --patience 3 \
    --holdout agents/move_history_branch --value-games data/fow_games
# after an interruption, continue from the last finished round:
python baselines/fow_games/rl_loop.py --out runs/rl_tf --resume --rounds 40 --patience 3 --holdout agents/move_history_branch
```

This produced `agents/turn_transformer_rl` (round 8: 1519 against the bots, 62.6% head to head
against `turn_transformer`; see its model card).

- `--patience 3` stops after 3 ratings in a row without a new best.
- `--holdout` also plays 200 games against an agent the loop never trains against at every
  rating (`holdout_score` in `log.jsonl`): if the bot rating rises while that score falls, the
  model is learning the bots rather than the game.
- `--resume` uses the run's saved settings (except `--rounds`, `--workers`, `--patience`,
  `--holdout`), rebuilds the buffer from the saved games and keeps the value network.
- The loop runs under `baselines/guard.py`: it waits for any other guarded job to finish, pauses
  while free memory is below 3 GB, memory use is above 85% or the GPU is above 83 °C, and lowers
  the number of game workers to what memory allows (about 0.7 GB each). Settings:
  `GUARD_MIN_FREE_GB`, `GUARD_MAX_RAM_PCT`, `GUARD_MAX_GPU_C`; `GUARD_OFF=1` turns it off.

Each round plays 6,000 new games, with game seeds no other round or arena run uses:
- 75% against the three bots;
- 25% against the last 3 saved versions of the model, the starting model included.

The learner samples every move at temperature 0.5 and reads the game turn by turn exactly as
`agent.py` does. In the first round the script checks that the training computation reproduces
the probabilities used in play; the smoke test matched to within 5e-6.

- **Buffer:** the last 2 rounds of games, counted in games. Each game is trained on twice per
  round and is then dropped, so no game is learned more than 4 times.
- **Long games:** a turn counts `min(1, 40 / turns in its game)`, so a long draw counts no more
  than a 40-turn game.
- **Value (critic):**
  - It is a copy of the policy's CNN and transformer, with its own head that also sees the
    opponent. It is used only in training and is not part of the agent.
  - It is warmed up on games already played (`--value-games`).
  - Every round it is first scored on the new games, which it has not seen, then trained on the
    buffer.
- **Advantages:** generalized advantage estimation over each game's turns, with λ = 0.8.
- **Policy update:**
  - Clipped policy gradient, with the ratio taken against the model that played the game and
    clip 0.2.
  - Plus 0.1 × KL(start ‖ current), to keep it near the imitation model.
  - Learning rate 5e-5, with dropout and batch-norm statistics frozen.
- **Choosing the round to keep:**
  - Every 2 rounds, the round is rated against the three bots with `fow.arena` on seed 9, which
    is not a training seed. The best-rated round is copied to `best/`.
  - Confirm the final pick on other seeds (6 and 7): choosing the best of several noisy ratings
    flatters the winner.

### Starting from the positions it loses (`--midgame-lost`)

```bash
python baselines/fow_games/rl_loop.py --init agents/turn_transformer_rl --out runs/rl_late --rounds 20 --patience 3     --seed 8 --midgame-share 0.5 --midgame-lost --midgame-plies 20,60 --midgame-games "runs/rl_tf/games/games_*.jsonl.gz"     --bot-weights 1,1,2 --value-init runs/rl_tf/value.safetensors --holdout agents/move_history_branch
```

- `--midgame-share 0.5`: half the games start from a recorded game in `--midgame-games` instead of
  move 1. The recorded moves are played through both players, so the model and its opponent build
  their memory of the game, but only the moves after them are learned from.
- `--midgame-lost`: only games the agent lost by having its king captured, cut 10 to 30 plies
  before the end (not before the first number of `--midgame-plies`). The model plays the losing
  side, against the bot that won. Without it, any long game is cut at a ply in `--midgame-plies`.
- `--bot-weights 1,1,2`: shares of greedy, guess-sf-4x500 and guess-sf-16x1000 among the bot
  games (here half against the bot it loses to most).
- `--value-init`: start the value network from a previous run's instead of warming one up.

This produced `agents/turn_transformer_late` (round 4: 1580 against the bots on seeds 6 and 7,
60.75% head to head against `turn_transformer_rl`, 78 losses in 1,200 bot games against 108).

Output: `log.jsonl` (one line per round: scores while sampling, draws, value error on new games,
KL to the start, share of clipped ratios, rating), `games/` (every game in `play_games.py`'s
format, reusable), `rounds/rNNN/` (each round as an agent directory), `best/` and
`value.safetensors`.

## Shard columns

One row per agent turn, ordered by game and then turn. See the docstring of `encode_games.py`:
`x`, `move`, `legal`, `logp`, `game`, `ply`, `turn`, `plies_left`, `result` (1 / 0 / -1 for the
agent), `opponent`, `teacher` (-1 when not labelled) and `opponents`.

Checked on a pilot: recomputing the log-probability of every recorded move from the re-encoded
inputs matches the value recorded in play to within 4e-6.
