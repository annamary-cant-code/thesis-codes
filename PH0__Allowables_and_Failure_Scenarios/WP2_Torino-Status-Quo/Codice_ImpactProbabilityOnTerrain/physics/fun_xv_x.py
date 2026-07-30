import numpy as np

def fun_xv_x(t, Vx_i, m, c):
    """Horizontal distance traveled up to time t under drag.
    x(t) = m/c * log(1 + Vx_i * c * t / m)
    """
    return m / c * np.log(1 + Vx_i * c * t / m)