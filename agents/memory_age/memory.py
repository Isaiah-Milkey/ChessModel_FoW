"""Memory that also records how long ago each remembered enemy piece was seen.

Built on the course's fow.memory.Memory, which keeps last_seen (square -> piece letter) and the
capture counts. This adds seen_ply: square -> the ply at which that piece was last seen there,
so features.py can fade old sightings instead of treating them all as equally certain.
"""
from fow.memory import Memory


class AgedMemory(Memory):
    def __init__(self) -> None:
        super().__init__()
        self.seen_ply: dict[str, int] = {}
        self.ply = 0

    def update(self, obs: dict) -> None:
        super().update(obs)
        self.ply = obs["ply"]
        for sq in obs["enemy"]:
            self.seen_ply[sq] = self.ply
        self.seen_ply = {s: p for s, p in self.seen_ply.items() if s in self.last_seen}

    def age(self, sq: str) -> int:
        """Own turns since the piece remembered on sq was last seen there (0 = in view now)."""
        return (self.ply - self.seen_ply[sq]) // 2
