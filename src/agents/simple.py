import random
from .base import *

class SimpleAgent(Agent):
    a: int = 0 # own attribute
    u: int = 0 # sought attribute
    ua_threshold: float = 0.0 # acceptance threshold
    w_threshold: float = 0.0 # threshold for willingness to date
    w_decrease_multiplier: int = 1 # multiplier to decrease willingness on rejection

    def __init__(self, id: int, a: int = random.randint(0, 50), u: int = random.randint(0, 50), gender: GENDER = "M", w: WILLINGNESS = 1.0, d: bool = False, e: bool = False) -> None:
        super().__init__(id, a, u, gender, w, d, e)
        self.ua_threshold =  random.uniform(0.0, 10.0)  # random acceptance threshold between 0 and 10
        self.w_threshold = random.uniform(0.0, 1)  # random willingness threshold between 0 and 1
        self.w_decrease_multiplier = random.randint(1, 100)  # random multiplier between 1 and 10

    def decide_match(self, other: "Agent") -> bool:
        """Decide whether to accept a match based on attribute similarity and willingness."""
        if self.w is None or other.w is None:
            return False  # If willingness is not defined, reject by default
        attribute_match = abs(self.a - other.u) < self.ua_threshold
        willingness_check = self.w >= self.w_threshold
        return attribute_match and willingness_check
    
    def handle_rejection(self, other: "Agent") -> None:
        """Handle rejection by slightly decreasing willingness."""
        if self.w is not None:
            self.w = max(0.0, self.w - 0.01 * self.w_decrease_multiplier)  # Decrease willingness by 0.01 * multiplier, not going below 0.0