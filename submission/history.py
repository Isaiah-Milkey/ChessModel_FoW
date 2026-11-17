"""The course's memory (fow.memory.Memory), plus the agent's own last few moves.

The observation's "last_move" is the move this side played on its previous turn, so the
history is built only from the agent's own observations.
"""
from fow.memory import Memory

N_MOVES_KEPT = 3


class HistoryMemory(Memory):
    def __init__(self) -> None:
        super().__init__()
        self.own_moves: list[str] = []   # most recent first, at most N_MOVES_KEPT

    def update(self, obs: dict) -> None:
        super().update(obs)
        mv = obs.get("last_move")
        if mv is not None:
            self.own_moves = [mv] + self.own_moves[:N_MOVES_KEPT - 1]
