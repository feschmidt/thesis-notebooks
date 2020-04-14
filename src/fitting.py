# Extracting resonance fequency and S11 line by line from raw dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stlabutils
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline


def func_PdBmtoV(PindBm, Z0=50):
    # converts powers in dBm to Volts
    return np.sqrt(2 * Z0 * 10**((PindBm - 30) / 10))


def func_PtoV(Vpeak, Z0=50):
    # converts volts to powers (both linear)
    return abs(Vpeak)**2 / 2 / Z0


def S11_fitting_full(data,
                     fitwidth=None,
                     trimwidth=5,
                     ftype='A',
                     doplots=False,
                     margin=51,
                     xvariable='Is (A)',
                     savefits=False,
                     data_offset=None):
    # #************************************************
    # fitwidth = None  # Width of trace to fit in units of detected peak width
    # trimwidth = 5.  # Width to trim around detected peak for background fit in units of detected peak width
    # ftype = 'A'  # A and -A are reflection models while B and -B are side coupled models
    # doplots = False  # Show debugging plots
    # margin = 51  # smoothing margin for peak detection
    # #************************************************

    mydfs = []

    S11res_list = []

    for myline in data:
        xval = myline[xvariable][0]
        print('##########################')
        print('xvariable:', xval)
        print('##########################')
        freqs = myline['Frequency (Hz)']
        S11complex = np.asarray([
            a + 1j * b for a, b in zip(myline['S21re ()'], myline['S21im ()'])
        ])  # S11 measurement
        if data_offset is not None:
            S11complex = S11complex / data_offset
        # Do fit with some given parameters.  More options available.
        params, _, _, _ = stlabutils.S11fit(freqs,
                                            S11complex,
                                            ftype=ftype,
                                            doplots=doplots,
                                            trimwidth=trimwidth,
                                            fitwidth=fitwidth,
                                            margin=margin)
        fitresnobg = stlabutils.utils.S11fit.S11theo(
            freqs, params, ftype=ftype)  # S11 remove background
        fitres = stlabutils.S11func(freqs, params,
                                    ftype=ftype)  # S11 include background

        f0 = params['f0'].value
        Qint = params['Qint'].value
        Qext = params['Qext'].value
        theta = params['theta'].value

        # Part1: S11(f) vs I0 (xval)
        mydict = {
            'Frequency (Hz)': freqs,
            xvariable: xval,
            'f0 (Hz)': f0,
            'Qint ()': Qint,
            'Qext ()': Qext,
            'theta (rad)': theta,
            'S11 ()': S11complex,
            'S11phase (rad)': np.angle(S11complex),
            'S11re ()': S11complex.real,
            'S11im ()': S11complex.imag,
            'S11abs ()': np.abs(S11complex),
            'S11fit ()': fitres,
            'S11fitphase (rad)': np.angle(fitres),
            'S11fitre ()': fitres.real,
            'S11fitim ()': fitres.imag,
            'S11fitabs ()': np.abs(fitres),
            'S11fitnobg ()': fitresnobg,
            'S11fitnobgphase (rad)': np.angle(fitresnobg),
            'S11fitnobgre ()': fitresnobg.real,
            'S11fitnobgim ()': fitresnobg.imag,
            'S11fitnobgabs ()': np.abs(fitresnobg)
        }
        mydfs.append(pd.DataFrame(mydict))

        if savefits:
            plt.plot(mydict['Frequency (Hz)'], mydict['S11abs ()'])
            plt.plot(mydict['Frequency (Hz)'], mydict['S11fitabs ()'])
            plt.title(xval)
            plt.show()
            plt.close()

        # Part2: Evaluating S11 on resonance
        try:
            S11_int = interp1d(freqs, S11complex)
            S11fit_int = interp1d(freqs, fitres)
            S11fitnobg_int = interp1d(freqs, fitresnobg)

            S11res_list.append({
                xvariable:
                xval,
                "f0 (Hz)":
                f0,
                "S11res ()":
                complex(S11_int(f0)),
                "S11res_fit ()":
                complex(S11fit_int(f0)),
                "S11res_fit_nobg ()":
                complex(S11fitnobg_int(f0)),
                'fitpars':
                params
            })

        except ValueError:  # dirty trick to not having to deal with interpolation out of range
            S11_int = InterpolatedUnivariateSpline(freqs, S11complex)
            S11fit_int = InterpolatedUnivariateSpline(freqs, fitres)
            S11fitnobg_int = InterpolatedUnivariateSpline(freqs, fitresnobg)

            S11res_list.append({
                xvariable:
                xval,
                "f0 (Hz)":
                f0,
                "S11res ()":
                complex(S11_int(f0)),
                "S11res_fit ()":
                complex(S11fit_int(f0)),
                "S11res_fit_nobg ()":
                complex(S11fitnobg_int(f0)),
                'fitpars':
                params
            })

    ds = S11res_list
    d = {}
    for k in ds[0].keys():
        d[k] = tuple(d[k] for d in ds)
    S11res_vs_I = pd.DataFrame(d)

    return S11res_vs_I, mydfs
