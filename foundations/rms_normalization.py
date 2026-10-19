import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x)
        gamma = np.array(gamma)
        rms = np.sqrt(np.sum(x**2 + 1e-5) / len(x))
        x_hat = x / rms
        output = x_hat * gamma
        return np.round(output, 4)