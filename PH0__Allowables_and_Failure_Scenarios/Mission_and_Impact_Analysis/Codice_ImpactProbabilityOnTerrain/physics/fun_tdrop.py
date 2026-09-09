import numpy as np

def fun_tdrop(m, c, y, Gd, Hd, Gamma, g):
    """Time to fall from peak altitude to ground."""
    return (np.arccosh(np.exp(c * y / m + Gd)) - Hd) * Gamma / g