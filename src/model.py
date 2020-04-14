"""
Model for current bias cavity terminated by Josephson junctions
"""

import numpy as np
from numpy import pi
from scipy.constants import e, Planck, k
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

echarge = e
Phi0 = Planck / (2 * echarge)
Rk = Planck / (echarge**2)
kB = k


def lossrate(x, k0, alpha, ix):
    # exponential dependence of loss rates on bias current
    return k0 + alpha * np.exp(abs(x) / ix)


def Lj(I, Ic):
    # Josephson inductance assuming sinusoidal CPR
    return Phi0 / (2 * pi * np.sqrt(Ic**2 - I**2))


def f0(I, fr, Lr, Ic, nJJ=1):
    # each JJ shifts the resonance frequency further downwards
    return fr / 2 * (1 + 1 / (1 + nJJ * Lj(I, Ic) / (Lr / 2)))


def Icmodel(f0, fr, Lr, I=0, nJJ=1):
    # calculate Ic based on resonance shift f0
    return np.sqrt(1 / ((Lr / (2 * f0 / fr - 1) - Lr) * pi / Phi0)**2 + I**2)


def Anh_seriesLCJ(Cr, Lr, Ic, I=0, n=1):
    # anharmonicity of a series C-L-nxLj circuit
    # source: calculated manually
    p = Lj(I, Ic) / (Lr + n * Lj(I, Ic))
    return -echarge**2 / (2 * Cr) * n * p**3


def Anh_Zhou(fr, Lr, Ic, I=0, n=1):
    # source: https://arxiv.org/pdf/1409.5630.pdf, https://arxiv.org/pdf/1006.2540.pdf
    p = Lj(I, Ic) / (Lr + n * Lj(I, Ic))
    w0 = 2 * pi * f0(I, fr, Lr, Ic, n)
    return -w0 * p**3 / (2 * n)


def Anh_Pogo(fr, Lr, Ic, I=0, n=1, Zr=50):
    # source: https://arxiv.org/pdf/1302.3484.pdf, eq. 26
    """p = Lj(I, Ic)/(Lr+n*Lj(I, Ic))
    w0 = 2*pi*f0(I, fr, Lr, Ic, n)
    return - pi**2 * w0 * Zr / Rk * p**3"""
    p = Lr / Lj(I, Ic)
    w0 = 2 * pi * f0(I, fr, Lr, Ic, n)
    return -pi**2 * Z0 * w0 / (16 * Rk / 2) * p**3


def Xpp(vph, Z0, l, Lj, w0):
    # source: general model, originating from Black Box Quantization (BBQ)
    numerator = 2 * echarge**2 * Lj * w0**2
    denominator = (1 + Lj * w0**2 * l /
                   (vph * Z0 * np.sin(l * w0 / vph)**2))**2
    return numerator / denominator


def CPRskew(tau=0):
    # calculate skew as a function of transmission
    phase = np.linspace(0, np.pi, 801)
    curr0 = CPR(phase, tau=0, norm=True)
    curr = CPR(phase, tau=tau, norm=True)
    idx = np.argmax(curr)
    phimax = phase[idx]
    phi0 = pi / 2
    S = phimax / phi0 - 1  # https://arxiv.org/pdf/1612.06895.pdf
    return S


def CPRtau(skew=0):
    # calculate transmission as a function of skew
    # Warning: slow
    taus = np.linspace(0, 1, 101)
    S = [CPRskew(tau) for tau in taus]
    return interp1d(S, taus)(skew)


def CPR(phase,
        tau=0, # transmission
        T=1e-3, # in K
        Rn=100, # in Ohm
        Delta=1e-3, # in V
        Ic=1e-6, # in A
        norm=True,
        **kwargs):
    # temperature effects are negligible for T<=1K
    Delta = Delta*echarge
    if tau == 0:
        full = Ic * np.sin(phase)
        if norm:
            return full / max(full)
        else:
            return full
    else:
        # short junction model
        # for Delta/(kB T) >> 1, the tanh() evaluates to 1
        # https://www.nature.com/articles/ncomms7181.pdf, Eq. 1
        # https://ris.utwente.nl/ws/portalfiles/portal/6702485/current-phase.pdf, Eq. 15
        prefactor = pi * Delta / (2 * echarge * Rn)
        sqrrt = np.sqrt(1 - tau * np.sin(phase / 2)**2)
        full = prefactor * np.sin(phase) / sqrrt * np.tanh(Delta /
                                                           (kB * T) * sqrrt)
        if norm:
            return full / max(full)
        else:
            return full


def Lj_of_I(I, Ic, tau=0, **kwargs):
    # https://www.nature.com/articles/s41467-018-06595-2.pdf, Eq. 1
    phase = np.linspace(-np.pi, np.pi, 201)
    curr = CPR(phase, tau=tau, norm=True, **kwargs)
    dIdphi = np.gradient(curr, phase)
    idx = dIdphi > 0
    if len(curr[idx]) == 0:
        phase = np.linspace(-0.999 * np.pi, 0.999 * np.pi, 201)
        curr = CPR(phase, tau=tau, norm=True, **kwargs)
        dIdphi = np.gradient(curr, phase)
        idx = dIdphi > 0
        if len(curr[idx]) == 0:
            print(tau)
            raise ValueError(
                'no positive dIdphi, even after reducing phase range')
    intfun = interp1d(curr[idx] * Ic, Phi0 / (2 * pi * dIdphi[idx] * Ic))
    try:
        return intfun(I)
    except ValueError:
        print('There was a ValueError')
        return InterpolatedUnivariateSpline(curr[idx] * Ic, Phi0 / (2 * pi * dIdphi[idx] * Ic))(I)
        #print('interpolation range:', intfun.x.min(), intfun.x.max())
        #print('x_new range:', I.min(), I.max())
        #raise ValueError(
        #    'A value in mycurr is outside the interpolation range.')


def f0_of_I(fr, Lr, I, Ic, tau=0, **kwargs):
    #return fr / (1 + Lj_of_I(I, Ic, tau, **kwargs) / Lr)
    return fr / 2 * (1 + 1 / (1 + Lj_of_I(I, Ic, tau, **kwargs) / (Lr / 2)))


def f0_of_Ic(fr, Lr, I, Ic, tau=0, **kwargs):
    return np.array(
        [f0_of_I(fr=fr, Lr=Lr, Ic=ic, tau=tau, I=0, **kwargs) for ic in Ic])


def Pogo_analytical(w, LJ, wr, Lr):
    # Pogorzalek analytical
    return Lr / LJ - pi / (2 * wr) * w * np.tan(pi / (2 * wr) * w)


def Pogo_approx(I, fr, Lr, Ic, nJJ=1):
    # each JJ shifts the resonance frequency further downwards
    return fr / (1 + nJJ * Lj(I, Ic) / Lr)


def Pogo_Icmodel(f0, fr, Lr, I=0, nJJ=1):
    # calculate Ic based on resonance shift f0
    return np.sqrt(I**2 + (Phi0 / (2 * pi * Lr) * 1 / (fr / f0 - 1))**2)


def gJJ_analytical(w, LJ, Cs, wr):
    # gJJ analytical
    Z0 = 50
    Z2 = 1 / (1j * w * Cs + 1 / Z0)
    Z1 = Z0 * (Z2 + 1j * Z0 * np.tan(pi * w / wr)) / (
        Z0 + 1j * Z2 * np.tan(pi * w / wr))
    ZJ = 1j * w * LJ
    Zq = 1 / (1 / ZJ + 1 / Z1)
    Yq = 1 / Zq
    return np.imag(Yq)


def gJJ_analytical_simple(w, LJ, wr):
    # gJJ approximation
    Z0 = 50
    lhs = w * LJ / Z0
    rhs = np.tan(pi * w / wr)
    return lhs - rhs
