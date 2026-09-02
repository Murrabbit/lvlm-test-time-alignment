import math
from collections import Counter
import numpy as np

def repeat_ratio(tokens, n=3):
    if len(tokens) < n:
        return 0.0

    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    total = len(grams)

    cnt = Counter(grams)
    repeats = sum(max(0, v - 1) for v in cnt.values())  

    smooth_term = math.ceil(max(10, total * 0.1))
    return (repeats + smooth_term) / (total + smooth_term)


def length_reward(x, L=1.0, a=0.05, b=0.8):
    return L * (1.0 - np.exp(-a * np.power(x, b)))