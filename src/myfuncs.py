# Auxiliary functions

import numpy as np


def func_PdBmtoV(PindBm, Z0=50):
    return np.sqrt(2 * Z0 * 10**((PindBm - 30) / 10))
