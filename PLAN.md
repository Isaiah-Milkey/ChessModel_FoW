# Project plan (proposal)

> Fill this in and submit your repository's link on Canvas by **Fri Oct 23, 11:59 PM**.
> Keep it short: about one page. You may update the plan later; record changes in `JOURNAL.md`.

- **Name:** Isaiah Milkey
- **GitHub username:** Isaiah-Milkey
- **Repository is private, with @3rdAT and @Slash0BZ as collaborators:** yes
## 1. The baseline, reproduced

What you ran (commands) and what you got. For example:
`python -m fow.arena agents/baseline bot:greedy bot:guess-sf-4x500 --games 100`

Ran locally (Windows, RTX 3060 Ti, Stockfish 19, seed 0):
`python -m fow.arena agents/baseline bot:greedy bot:guess-sf-4x500 --games 100 --device cuda --workers 4`

| Opponent | Score | Notes |
|---|---|---|
| bot:greedy | 77.0 / 100 | all 200 games ended by king capture |
| bot:guess-sf-4x500 | 78.0 / 100 | |

Local rating of the baseline: **1246** (95% interval 1176–1316), consistent with the published 1217.

## 2. What you will try first

Two or three concrete ideas. For each one:

### Idea 1:
- **What:** the change. 
    I plan to add a memory block to help the model remember where seen pieces are on the board.
- **Why you expect it to help:**
    This should help the model make more informed decisions.
- **How you will measure it:** which matches, how many games, compared against what.
    I will compare this to the baseline model and see how many games it wins compared to the standard bot against the stockfish trained one.

### Idea 2:
- **What:**
    Grow the model and training scale.
- **Why you expect it to help:**
    The model currently is under the max limit of parameters, so it is able to be scaled up. Larger models with more data generally performs better.
- **How you will measure it:**
    I will compare this to the baseline model and see how many games it wins compared to the standard bot against the stockfish trained one.

### Idea 3: sequential learning (added 11/7/26, drafted with Claude Code)
- **What:**
    I want to implement sequential learning: the model should learn from games in order, not from
    random snapshots. Taking random screenshots of games is really bad. Right now each game is cut
    into random turns (a coin flip per turn), all turns from all games are shuffled together, and
    the model only sees one turn at a time. The only history it gets is what we summarise for it
    (memory of last-seen pieces, and in `move_history` the turn number and its own last 3 moves).
    Plan, in two steps:
    1. Snapshot history: every input also contains what the agent saw 1, 2 and 3 turns ago (like
       AlphaZero's history planes). A new `features.py` only; the recorded games (`--record`) can
       be re-encoded with `reencode.py`, so no new Stockfish labels are needed.
    2. A network with memory (LSTM, GRU or a small transformer) that reads the game turn by turn
       and keeps its own running memory, trained on whole games with the loss on the labelled
       turns. The agent carries that memory from one turn to the next.
- **Why you expect it to help:**
    Turn order matters: the way a game goes creates patterns (what moved, what appeared and
    disappeared, what the opponent seems to be doing). Like GPT predicting the next token from the
    tokens before it, the model should predict the next move from the sequence of turns before
    it, not from the current board alone. Nothing so far shows the model what the opponent did.
- **How you will measure it:**
    Same recorded games and teacher labels as the model it is compared with (`move_history` with
    dropout 0.1), validation on held-out views it never trained on (`train.py --split view`), then
    200 games each against bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000 on the same
    arena seed, plus 1,000 games head to head.

### Idea 4: fine-tune on games against the FoW bots (added 11/7/26, drafted with Claude Code)
- **What:**
    Start from the best model so far (the branching-games model, `runs/mh_drop10_branch`, 1401)
    and fine-tune it on games it plays against the Fog-of-War bots (bot:greedy,
    bot:guess-sf-4x500, bot:guess-sf-16x1000), instead of only copying the teacher. The model
    learns from how its own games turned out: the moves it played in games it won are made more
    likely, and the moves in games it lost less likely. During these games it samples its moves,
    so it tries alternatives. A small learning rate and a penalty for drifting far from the
    starting model keep it from forgetting what it learned from the teacher.
- **Why you expect it to help:**
    We learned that training strictly to the teacher is not good. Stockfish sees the whole board,
    so it often asks for moves the agent cannot justify from what it sees, and matching it more
    often does not mean playing better: the branching-games model matched the teacher less often
    (40.4% vs 41.3% on the old games) but beat the reference 653 / 1000 (+110 Elo). It got better
    by learning from the positions it actually reaches in its own games. Learning from the results
    of games played in the fog goes one step further: the signal comes from winning in Fog of War,
    not from a teacher playing regular chess.
- **How you will measure it:**
    The training games use different seeds from the evaluation games. Then 200 games each against
    bot:greedy, bot:guess-sf-4x500 and bot:guess-sf-16x1000 on arena seed 6 (the same 600 games
    as every model since `move_history`), plus 1,000 games head to head against the
    branching-games model it started from. Also watch that it does not just learn to exploit one
    bot: compare the score against each bot separately.

## 3. Compute plan

What you expect to run on Sol: data generation, training runs, rough GPU-hours. How your final
pipeline will stay reproducible in hours, not days.

Data generation will be reproducible and stored on the Github so a dataset can be generated and reused for later training. Training will be done on SOL until the model flattens in performance.

## 4. Risks and questions

What might not work, and what you would do instead. Questions for the staff.

Since fog of war is a more 'random' style of chess, I belive luck could cause the model to learn the wrong types of gameplay. Fixing this could require specific styles of learning that rewards a more strategic play style. No quesitons for staff.

# 11/6/26 (Written by Isaiah)
What I've learned:
- Scaling up the model doesn't necessarily help? The baseline model size is large enough to learn given current training data. Larger model is overkill for no significant change. This is supported from experiments.
- Training on more games helps increase to about 1360 elo. This is where the model plateus. Larger models just take longer to train, but I dont think this model is that large.
- Memory doesnt update until the piece has been seen again. I bet this leads to lots of misinformed play. (Should add a degradation of memory over time).
- Model learns from random sets of turns independent of time or previous turns. I think this is bad as turn count and previous moves are important? Can we have a memory block with turn number being a factor?
- Training is learned from Stockfish knowledge, meaning its learning from a bot playing a different game (not FoW).

What this tells me:
- This reminds me of BERT vs GPT. Bert was bad at generating good writing because it was trained to see bidirectionally. Chess is similar where the model is learning from a bot that has access to the whole board even though it only knows its own position.
GPT came out on top because of scaling parameter size, and training the next token based off of previous tokens.
I want to do something similar where it predicts next play off of past sequence of turns- rather than current board= best play. Context matters.

Things to add/change.
Have memory deteriorate over moves (3/4 turns?)
Add training for sequential plays (to remember patterns/games)
Train on FoW games?

Generated a site to learn the project better: [link](https://claude.ai/artifact/DR62zpnhh3Chg1APMBaQmJ#how)

# 11/7/26 (drafted with Claude Code)
What we learned:
- Strict training to the teacher is not good. Stockfish plays with the whole board visible, so
  copying it more closely does not make a better Fog-of-War player.
- Dropout and merging repeated positions barely changed anything. New, varied games that the
  agent played itself, deeper into the game, gave the first clear gain since more data:
  1401 Elo, and 653 / 1000 head to head against the same model trained on the old games.

Next: fine-tune the best model on games against the FoW bots, learning from the game results
(Idea 4).