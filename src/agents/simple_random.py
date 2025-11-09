import random
import numpy as np
from .base import *

class SimpleRandomAgent(Agent):
    W_THRESHOLD: int = 0 # threshold for willingness to date
    W_MULTIPLIER: int = 1 # multiplier to decrease willingness on rejection
    W_START: int = 50
    A_MAX: float = 10.0
    A_MIN = float = - 10.0

    def __init__(self, id: int, a: int, u: int, gender: GENDER = "M", w: WILLINGNESS = 50, d: bool = False, e: bool = False) -> None:
        super().__init__(id, a, u, gender, w, d, e)
        self.w = self.W_START

    def decide_match(self, other: Agent) -> bool:
        """Decide whether to accept a match based on attribute similarity and willingness."""
        if self.w is None or other.w is None:
            return False  # If willingness is not defined, reject by default
        match_probability = (self.A_MAX - abs(self.u - other.a))/self.A_MAX
        attribute_check = np.random.binomial(1, match_probability)
        willingness_check = self.w >= self.W_THRESHOLD

        return bool(attribute_check and willingness_check)
    
    def handle_rejection(self, other: Agent) -> None:
        """Handle rejection by slightly decreasing willingness."""
        if self.w is not None:
            self.w = max(0, self.w - self.W_MULTIPLIER)  # Decrease willingness by 0.01 * multiplier, not going below 0.0
