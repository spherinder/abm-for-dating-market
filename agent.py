from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Agent:
    a: Optional[int] = None # own attribute
    u: Optional[int] = None # sought attribute

    def __init__(self, a: Optional[int] = None, u: Optional[int] = None) -> None:
        self.a = a
        self.u = u

    def update(self, a: Any = None, s: Any = None) -> None:
        """Update attributes a and/or s (only non-None values are applied)."""
        if a is not None:
            self.a = a
        if s is not None:
            self.u = s

    def to_dict(self) -> dict:
        """Return a dictionary representation of the agent."""
        return {"a": self.a, "s": self.u}

    def __repr__(self) -> str:
        return f"Agent(a={self.a!r}, s={self.u!r})"