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

## 3. Compute plan

What you expect to run on Sol: data generation, training runs, rough GPU-hours. How your final
pipeline will stay reproducible in hours, not days.

Data generation will be reproducible and stored on the Github so a dataset can be generated and reused for later training. Training will be done on SOL until the model flattens in performance.

## 4. Risks and questions

What might not work, and what you would do instead. Questions for the staff.

Since fog of war is a more 'random' style of chess, I belive luck could cause the model to learn the wrong types of gameplay. Fixing this could require specific styles of learning that rewards a more strategic play style. No quesitons for staff.
