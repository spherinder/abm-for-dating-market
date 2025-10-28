from dataclasses import dataclass
from typing import Any, Optional, Literal

GENDER = Literal["M", "F", "O", None]
WILLINGNESS = Optional[float]  # willingness to date (0.0 to 1.0)

@dataclass
class Agent:
    a: Any | None = None # own attribute
    u: Any | None = None # sought attribute
    g: GENDER = None # gender 
    w: WILLINGNESS = None # willingness to date (0.0 to 1.0)
    d: bool = False # is dating
    e: bool = False # is excluded from dating
    def __init__(self, a: Any | None = None, u: Any | None = None, g: GENDER = "M", w: WILLINGNESS = None, d: bool = False, e: bool = False) -> None:
        self.a = a # own attribute
        self.u = u # sought attribute
        self.g = g # gender
        self.w = w # willingness to date (0.0 to 1.0)
        self.d = d # is dating
        self.e = e # is excluded from dating

    def decide_match(self, other: "Agent") -> bool:
        raise NotImplementedError("Subclasses should implement this method.")

    def update(self, a: Any = None, s: Any = None, gender: GENDER = "M", w: float | None= None) -> None:
        """Update attributes a and/or s (only non-None values are applied)."""
        if a is not None:
            self.a = a
        if s is not None:
            self.u = s
        if w is not None:
            self.w = w
        self.g = gender

    def to_dict(self) -> dict:
        """Return a dictionary representation of the agent."""
        return {"a": self.a, "s": self.u, "g": self.g, "w": self.w}

    def __repr__(self) -> str:
        return f"Agent(a={self.a!r}, s={self.u!r}, g={self.g!r}, w={self.w!r})"
