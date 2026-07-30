import numpy as np

def fun_yu(m, c, g, gamma, t, Hu, Gu):
    """Altitude gained during upward phase up to time t."""
    return -m / c * (np.log(np.cos(g * gamma * t + Hu)) - Gu)