import random
from base import *

class SimpleAgent(Agent):
    a: int = None # own attribute
    u: int = None # sought attribute
    ua_threshold: float | None = None # acceptance threshold
    w_threshold: float | None = None # threshold for willingness to date
    def __init__(self, a: int = 1, u: int = 1, gender: GENDER = "M", w: WILLINGNESS = 1.0, d: bool = False, e: bool = False) -> None:
        super().__init__(a, u, gender, w, d, e)
        self.ua_threshold =  random.uniform(0.0, 10.0)  # random acceptance threshold between 0 and 10
        self.w_threshold = random.uniform(0.0, 0.9999)  # random willingness threshold between 0 and 1

    def decide_match(self, other: "Agent") -> bool:
        """Decide whether to accept a match based on attribute similarity and willingness."""
        if self.w is None or other.w is None:
            return False  # If willingness is not defined, reject by default
        attribute_match = abs(self.a - other.u) < self.ua_threshold
        willingness_check = self.w >= self.w_threshold
        return attribute_match and willingness_check