# Source code for error propagation calculations on Josephson inductance

import numpy as np


def J(f0, fr, Lr):
    nu = fr / f0
    return Lr * dJdLr(f0, fr)


def dJdLr(f0, fr):
    nu = fr / f0
    return (nu - 1) / (2 - nu)

def dJdnu(f0, fr, Lr):
    nu = fr / f0
    first = Lr / (2 - nu)
    second = -Lr * (nu - 1) / (2 - nu)**2
    return first + second

def dnudfr(f0, fr):
    return 1 / f0

def dnudf0(f0, fr):
    return -fr / f0**2

def dJdf0(f0, fr, Lr):
    return dJdnu(f0, fr, Lr) * dnudf0(f0, fr)

def dJdfr(f0, fr, Lr):
    return dJdnu(f0, fr, Lr) * dnudfr(f0, fr)

def sigmaLJ(f0, fr, Lr, sigmaf0, sigmafr, sigmaLr):
    return np.sqrt(
        dJdf0(f0, fr, Lr)**2 * sigmaf0**2 + dJdfr(f0, fr, Lr)**2 * sigmafr**2 +
        dJdLr(f0, fr)**2 * sigmaLr**2)
