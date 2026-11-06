# AI assistance disclosure

List every AI assistant or coding agent you used for this project. Keep it current.

| Tool | What you used it for | Roughly how much | Where (files, commits, journal entries) |
|---|---|---|---|
| Claude Code (Claude Opus 5.5) | 2026-10-24 setup session: explained the repo setup and how to make a private copy with the staff as collaborators; installed dependencies (`fow`, `chess`, `safetensors`); downloaded the Windows Stockfish 19 build; ran the baseline arena evaluation; drafted Idea 2 and the §1 baseline results in `PLAN.md`; drafted the first journal entry. I reviewed and edited everything it wrote. | One session, about 1 hour | `PLAN.md` (§1, Idea 2), `JOURNAL.md` (2026-10-24 entry), `AI_USE.md` |
| Claude Code (Claude Opus 5.5) | 2026-11-05 scaling session: regenerated the baseline data and generated a 20K-game dataset; trained and evaluated 64x4 and 128x4 models; found that deterministic network-vs-network games repeat; wrote the random-opening change in `agents/baseline_4x/agent.py`; wrote the `baseline_4x` model card; drafted the journal entries. | One session, about 2 hours | `agents/baseline_4x/`, `JOURNAL.md` (2026-11-05 entries), `AI_USE.md` |
| Claude Code (Claude Opus 5.5) | 2026-11-05/06 session: generated the 100K-game dataset and diagnosed the hung run; added `--save-best` to `train.py`; trained 64x4 and 128x4 to plateau and evaluated them; wrote the `baseline_75k` model card; made a visual explainer of the system; drafted the facts of the 2026-11-06 journal entry (the hypothesis, surprises and decision are my own). | One session, about 3 hours (mostly waiting on runs) | `baselines/imitation/train.py`, `agents/baseline_75k/`, `JOURNAL.md` (2026-11-06 entry), `AI_USE.md` |

Commits written mostly by an agent should carry a trailer naming the tool, for example
`Co-Authored-By: <tool name>`.
