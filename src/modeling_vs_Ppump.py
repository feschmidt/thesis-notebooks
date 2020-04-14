"""
This is the algorithm for evaluating the power-dependent cavity response for a predefined measurement set.
"""

import lmfit
from lmfit.models import ExpressionModel
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import os
import pandas as pd
import pickle
from scipy import signal
from scipy.constants import hbar
import stlabutils


def photon_number(P_p_lin, f_p, f_0, ki, ke):
    """
    Calculate photon number for fixed pump and low-power probe
    """
    Delta = f_0 - f_p
    wp = 2 * pi * f_p
    k = ki + ke
    return 4 * P_p_lin / (hbar * wp) * ke / (k**2 + 4 * Delta**2)


def photon_flux(P_p_lin, f_p):
    """
    Calculate on-chip photon flux
    """
    return P_p_lin / hbar / 2 / pi / f_p


def nonlinear_response(alpha_0, beta, P_p_lin, f_p, f_0, k, kext):
    """
    Calculate S11 response of Duffing oscillator
    """
    Delta = f_p - f_0
    n_p = photon_flux(P_p_lin, f_p)
    phi = np.arctan(-2 * (Delta - beta * alpha_0) / k)
    S11 = 1 - np.sqrt(2 * pi * kext * alpha_0) / np.sqrt(n_p) * np.exp(
        -1j * phi)
    return S11.flatten()


def solve_polynom(Delta, n_p, beta, k, kext):
    """
    Solve third-order polynom as solution to Duffing oscillator
    """
    a = 4 * pi**2 * beta**2
    b = -2 * 2 * pi * Delta * 2 * pi * beta
    c = (2 * pi * Delta)**2 + (2 * pi * k)**2 / 4
    d = -2 * pi * kext * n_p
    coeff = [a, b, c, d]
    return np.roots(coeff)


def calc_alpha0(beta, f_p, f_0, k, kext, P_p_lin, sol, doprint):
    """
    calculate intracavity field amplitude
    """
    Delta = f_p - f_0
    n_p = photon_flux(P_p_lin, f_p)
    alphas = []
    for i, (dd, nn) in enumerate(zip(Delta, n_p)):
        roots = solve_polynom(Delta[i], n_p[i], beta, k, kext)
        # Disregard roots with nonzero imaginary part
        roots = roots[np.imag(roots) == 0]
        # Choose branch:
        if sol == 'min':
            alpha_0 = np.min(roots)
        elif sol == 'max':
            alpha_0 = np.max(roots)
        elif sol == 'med':
            alpha_0 = np.median(roots)

        if doprint:
            print(i + 1, len(Delta),
                  f'{(i + 1) / len(Delta) * 100:.03f}% done')

        alphas.append(alpha_0)

    return np.array(alphas)


def power_VNA_to_chip(P_out_VNA, att_av):
    """
    Convert VNA power (dBm) to on-chip power by attenuation (dB)
    """
    P_p_log = P_out_VNA - abs(att_av)
    P_p_lin = 10**(P_p_log / 10) * 1e-3  # dBm to W
    return P_p_lin


def fitmodel(f_0,
             kint,
             kext,
             beta,
             att_av,
             sol='min',
             P_out_VNA=-30,
             Probe_freq=8e9,
             doprint=False):
    """
    this is the model we use for fitting the anharmonic cavity response

    - sol='min': find smallest real and positive solution for the low-amplitude branch
    - sol='max': find largest real and positive solution for the high-amplitude branch
    - sol='med': find intermediate branch value
    """

    f_p = Probe_freq
    k = kint + kext

    # On-chip Probe Power
    P_p_lin = power_VNA_to_chip(P_out_VNA, att_av)

    alpha_0 = calc_alpha0(beta, f_p, f_0, k, kext, P_p_lin, sol, doprint)

    # Calculate response
    S11 = nonlinear_response(alpha_0, beta, P_p_lin, f_p, f_0, k, kext)

    return S11


def kint_NL(Pin, k0, gamma):
    """
    Model for nonlinear internal loss rate growing with the square root of input power
    """
    return k0 * (gamma * np.sqrt(Pin) + 1)


def calc_alpha0NL(beta, f_p, f_0, kint, gamma, kext, P_p_lin, sol, doprint):
    """
    calculate intracavity field amplitude including nonlinear damping
    """
    Delta = f_p - f_0
    n_p = photon_flux(P_p_lin, f_p)
    K = kint_NL(P_p_lin, kint, gamma) + kext
    alphas = []
    for i, (dd, nn, k) in enumerate(zip(Delta, n_p, K)):
        roots = solve_polynom(Delta[i], n_p[i], beta, k, kext)
        # Disregard roots with nonzero imaginary part
        roots = roots[np.imag(roots) == 0]
        # Choose branch:
        if sol == 'min':
            alpha_0 = np.min(roots)
        elif sol == 'max':
            alpha_0 = np.max(roots)
        elif sol == 'med':
            alpha_0 = np.median(roots)

        if doprint:
            print(i + 1, len(Delta),
                  f'{(i + 1) / len(Delta) * 100:.03f}% done')

        alphas.append(alpha_0)

    return np.array(alphas)


def nonlinear_responseNL(alpha_0, beta, P_p_lin, f_p, f_0, kint, gamma, kext):
    """
    Calculate S11 response of Duffing oscillator
    """
    Delta = f_p - f_0
    n_p = photon_flux(P_p_lin, f_p)
    k = kint_NL(P_p_lin, kint, gamma) + kext
    phi = np.arctan(-2 * (Delta - beta * alpha_0) / k)
    S11 = 1 - np.sqrt(2 * pi * kext * alpha_0) / np.sqrt(n_p) * np.exp(
        -1j * phi)
    return S11.flatten()


def fitmodel_NL(f_0,
                kint,
                kext,
                beta,
                att_av,
                gamma,
                sol='min',
                P_out_VNA=-30,
                Probe_freq=8e9,
                doprint=False):
    """
    this is the model we use for fitting the anharmonic cavity response, assuming nonlinear damping

    # sol='min': find smallest real and positive solution for the low-amplitude branch
    # sol='max': find largest real and positive solution for the high-amplitude branch
    # sol='med': find intermediate branch value
    """

    f_p = Probe_freq

    # On-chip Probe Power
    P_p_lin = power_VNA_to_chip(P_out_VNA, att_av)

    # non-linear internal loss rate
    k = kint_NL(P_p_lin, kint, gamma) + kext

    alpha_0 = calc_alpha0NL(beta, f_p, f_0, kint, gamma, kext, P_p_lin, sol,
                            doprint)

    # Calculate response
    S11 = nonlinear_responseNL(alpha_0, beta, P_p_lin, f_p, f_0, kint, gamma,
                               kext)

    return S11


class MyWrapper():
    """
    Class to execute some of the above code to make it easier for analyzing many datafiles.
    """
    def __init__(self,
                 classparams,
                 ikint=0,
                 bgremoval=True,
                 trimming=True,
                 dev='2x2',
                 devdf=None):

        for key, val in classparams.items():
            setattr(self, key, val)

        measfile = self.measfile

        self.mymodel = ModelPowerDep({'measfile': measfile})

        self.mymodel.load_data(ikint=ikint,
                               bgremoval=bgremoval,
                               dev=dev,
                               devdf=devdf)

        self.X2, self.Y2, self.DAT2 = self.mymodel.inspect_measurement_data(
            trimming=trimming)

    def guess(self, beta, att_av, **kwargs):
        """
        Evaluate nonlinear model with given input parameters
        """

        self.X2, self.Y2, self.S11theo2 = self.mymodel.model_measurement_data(
            att_av, beta, sol='min', doplots=True, **kwargs)

    def execute_fit(self,
                    beta,
                    att_av,
                    betavary=True,
                    f0vary=True,
                    kintvary=True,
                    kextvary=True,
                    attavvary=False):
        """
        Run the fit
        """

        self.myresult, self.myfitdata = self.mymodel.fit_measurement_data(
            att_av,
            beta,
            sol='min',
            doplots=True,
            betavary=betavary,
            f0vary=f0vary,
            kintvary=kintvary,
            kextvary=kextvary,
            attavvary=attavvary)

        print(self.myresult.params.pretty_print())

    def visualize_fit(self):
        """
        Plot fit data, abs and phase
        """

        fig, ax = plt.subplots(constrained_layout=True)
        plt.axis('off')
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.pcolormesh(self.X2, self.Y2, np.abs(self.myfitdata))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('VNA power (dBm)')
        plt.title('Abs (Fit)')
        ax2 = fig.add_subplot(gs[0, 1])
        plt.pcolormesh(self.X2, self.Y2, np.angle(self.myfitdata))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('VNA power (dBm)')
        plt.title('Phase (Fit)')
        plt.show()
        plt.close()

        for i in [0, 10, 20, 30, 40]:
            fig, ax = plt.subplots(constrained_layout=True)
            plt.axis('off')
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            plt.plot(self.Y2[:, i], np.abs(self.DAT2[:, i]), 'o', label='data')
            plt.plot(self.Y2[:, i], np.abs(self.myfitdata[:, i]), label='fit')
            plt.legend()
            plt.title('Abs' + str(self.X2[0, i]))
            if i == 0:
                plt.ylim(ymax=1.02)
                firstlims = plt.gca().get_ylim()
            else:
                plt.ylim(firstlims)
            ax2 = fig.add_subplot(gs[0, 1])
            plt.plot(self.Y2[:, i],
                     np.angle(self.DAT2[:, i]),
                     'o',
                     label='data')
            plt.plot(self.Y2[:, i],
                     np.angle(self.myfitdata[:, i]),
                     label='fit')
            plt.legend()
            plt.title('Phase' + str(self.X2[0, i]))
            if i == 0:
                firstlims2 = plt.gca().get_ylim()
            else:
                plt.ylim(firstlims2)
            plt.show()
            plt.close()


class MyWrapperKint(MyWrapper):
    """
    Subclass of MyWrapper with the aim of allowing kint to vary in between pump powers
    """
    def __init__(self,
                 classparams,
                 mydf,
                 betavary=False,
                 f0vary=False,
                 kintvary=True,
                 kextvary=False,
                 attavvary=False,
                 **kwargs):

        super().__init__(classparams, **kwargs)

        Vg = self.mymodel.Vg
        self.Vg = Vg
        thebeta = mydf.loc[Vg]['beta (Hz)']
        thef0 = mydf.loc[Vg]['f0 (Hz)']
        theke = mydf.loc[Vg]['kext (Hz)']
        theki = mydf.loc[Vg]['kint (Hz)']
        theatt = mydf.loc[Vg]['att_av (dB)']
        self.theki = theki

        self.mymodel = lmfit.Model(
            fitmodel, independent_vars=['P_out_VNA', 'Probe_freq'])
        print(self.mymodel.param_names, self.mymodel.independent_vars)

        params = self.mymodel.make_params()
        params.add('beta',
                   value=thebeta,
                   min=-2 * abs(thebeta),
                   max=-1 / 2 * abs(thebeta),
                   vary=betavary)
        params.add('f_0',
                   value=thef0,
                   min=0.9 * thef0,
                   max=1.1 * thef0,
                   vary=f0vary)
        params.add(
            'kint',
            value=theki,
            min=0,
            #min=1.0 * theki,
            #max=1.2 * theki,
            vary=kintvary)
        params.add(
            'kext',
            value=theke,
            #min=0.8 * theke,
            #max=1.2 * theke,
            vary=kextvary)
        params.add('att_av',
                   value=theatt,
                   min=-1.2 * abs(theatt),
                   max=-0.8 * abs(theatt),
                   vary=attavvary)

        print(params.pretty_print())
        self.params = params

    def execute_fit(self, verbose=True, reuse=True):
        results = []
        for i, (x, y, z) in enumerate(zip(self.X2.T, self.Y2.T, self.DAT2.T)):
            if i == 0:
                result = self.mymodel.fit(z,
                                          P_out_VNA=x[0],
                                          Probe_freq=y,
                                          params=self.params)
            else:
                result = self.mymodel.fit(z,
                                          P_out_VNA=x[0],
                                          Probe_freq=y,
                                          params=result.params)

            if verbose:
                print(x[0])
                print(result.fit_report())
            results.append(result)

        self.kint = np.array([x.params['kint'].value for x in results])
        self.dkint = np.array([x.params['kint'].stderr for x in results])
        self.DAT2FIT = np.reshape([x.best_fit for x in results],
                                  self.DAT2.T.shape).T

    def visualize_fitpar(self):
        fig = plt.figure(constrained_layout=True)
        #gs = fig.add_gridspec(1, 2)
        #ax1 = fig.add_subplot(gs[0, 0])
        plt.ylabel('kint (Hz)')
        plt.errorbar(self.X2[0], self.kint, self.dkint, fmt='.-')
        plt.axhline(self.theki, c='C3')
        #ax2 = fig.add_subplot(gs[0, 1])
        #plt.ylabel('kext (Hz)')
        #plt.errorbar(P_out_VNA,[x.params['kext'].value for x in results],[x.params['kext'].stderr for x in results],fmt='.-')
        #plt.axhline(theke, c='C3')
        plt.show()
        plt.close()

    def visualize_fit(self):
        for i, (x, y, z, z2) in enumerate(
                zip(self.X2.T, self.Y2.T, self.DAT2.T, self.DAT2FIT.T)):
            #if i % 1 == 0:
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            plt.plot(y, np.abs(z), 'o')
            plt.plot(y, np.abs(z2))
            ax2 = fig.add_subplot(gs[0, 1])
            plt.plot(y, np.angle(z), 'o')
            plt.plot(y, np.angle(z2))
            plt.show()
            plt.close()


class MyWrapperKintNonlinear(MyWrapper):
    """
    Subclass of MyWrapper with the aim of introducing nonlinear kint as a function of input power
    """
    def __init__(self,
                 classparams,
                 mydf,
                 mygamma,
                 betavary=True,
                 f0vary=False,
                 kintvary=True,
                 kextvary=False,
                 attavvary=False,
                 gammavary=True,
                 **kwargs):

        super().__init__(classparams, **kwargs)

        Vg = self.mymodel.Vg
        self.Vg = Vg
        thebeta = mydf.loc[Vg]['beta (Hz)']
        thef0 = mydf.loc[Vg]['f0 (Hz)']
        theke = mydf.loc[Vg]['kext (Hz)']
        theki = mydf.loc[Vg]['kint (Hz)']
        theatt = mydf.loc[Vg]['att_av (dB)']
        thegamma = mygamma.loc[Vg]['gamma']

        self.mymodelNL = lmfit.Model(
            fitmodel_NL, independent_vars=['P_out_VNA', 'Probe_freq'])
        print(self.mymodelNL.param_names, self.mymodelNL.independent_vars)

        params = self.mymodelNL.make_params()
        params.add('gamma',
                   value=thegamma,
                   min=0.5 * thegamma,
                   max=2 * thegamma,
                   vary=gammavary)
        params.add('beta',
                   value=thebeta,
                   min=-2 * abs(thebeta),
                   max=-1 / 2 * abs(thebeta),
                   vary=betavary)
        params.add('f_0',
                   value=thef0,
                   min=0.9 * thef0,
                   max=1.1 * thef0,
                   vary=f0vary)
        params.add(
            'kint',
            value=theki,
            #min=0,
            min=1.0 * theki,
            max=1.2 * theki,
            vary=kintvary)
        params.add('kext',
                   value=theke,
                   min=0.8 * theke,
                   max=1.2 * theke,
                   vary=kextvary)
        params.add('att_av',
                   value=theatt,
                   min=-1.2 * abs(theatt),
                   max=-0.8 * abs(theatt),
                   vary=attavvary)

        print(params.pretty_print())
        self.params = params

        print(params.pretty_print())

    def execute_fit(self):
        result = self.mymodelNL.fit(self.DAT2.flatten(),
                                    P_out_VNA=self.X2.flatten(),
                                    Probe_freq=self.Y2.flatten(),
                                    params=self.params)

        print(result.fit_report())
        # reshape fitdata to do 2D plot
        fitdata = result.best_fit.reshape(self.DAT2.shape)
        self.fitdata = fitdata
        self.result = result

        return result, fitdata

    def visualize_fit(self):
        for i, (x, y, z, z2) in enumerate(
                zip(self.X2.T, self.Y2.T, self.DAT2.T, self.fitdata.T)):
            #if i % 1 == 0:
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            plt.plot(y, np.abs(z), 'o')
            plt.plot(y, np.abs(z2))
            ax2 = fig.add_subplot(gs[0, 1])
            plt.plot(y, np.angle(z), 'o')
            plt.plot(y, np.angle(z2))
            plt.show()
            plt.close()


class ModelPowerDep():
    """
    Class to load, plot and process VNA sweeps of nonlinear resonators as a function of probe power
    """
    def __init__(self, classparams):

        self.colnames = [
            'Probe Frequency (Hz)', 'S11re ()', 'S11im ()', 'S11dB (dB)',
            'S11ph (rad)', 'S11abs ()', 'Intracavity photons ()',
            'Probe power (W)', 'Probe power (dBm)', 'Beta (Hz)',
            'VNA power (dBm)', 'Vgate (V)'
        ]

        for key, val in classparams.items():
            setattr(self, key, val)

    def load_data(self, ikint=-1, bgremoval=True, dev='2x2', devdf=None):
        """
        Load measurement data, remove background, unwrap and detrend phase
        """

        measfile = self.measfile

        print('Loading in data')
        print(measfile)
        measdata = stlabutils.readdata.readdat(measfile)

        Vgmeas = measdata[0]['Vgate (V)'][0]
        Vgstring = f'{int(Vgmeas):+03d}'
        print('Vgstring:', Vgstring)

        print('Loading measdata\n')
        measbgRe = stlabutils.framearr_to_mtx(measdata,
                                              xkey='Frequency (Hz)',
                                              ykey='Power (dBm)',
                                              key='S21re ()')
        measbgIm = stlabutils.framearr_to_mtx(measdata,
                                              xkey='Frequency (Hz)',
                                              ykey='Power (dBm)',
                                              key='S21im ()')

        print('Unwrapping and detrending phase')
        measabs = np.abs(measbgRe.pmtx.values + 1j * measbgIm.pmtx.values)
        measphase = signal.detrend(
            np.unwrap(
                np.angle(measbgRe.pmtx.values + 1j * measbgIm.pmtx.values)))

        if bgremoval == True:
            print('\nRemoving background\n')
            BGnocavity = pickle.load(
                open('pkldump/2x2_processing_power_dep_Vg_background.pkl',
                     'rb'))
            S21offset = BGnocavity['S11 (mtx)']
            measclean = (measabs *
                         np.exp(1j * measphase)) / (S21offset.pmtx.values)
        elif bgremoval == False:
            print('\nSkipping background removal\n')
            measclean = (measabs * np.exp(1j * measphase))
        elif bgremoval == 'manual':
            print('\n Manual background removal\n')
            measclean = (measabs * np.exp(1j * measphase)) / (
                measabs[0, 0] * np.exp(1j * measphase[0, 0]))
        else:
            raise ValueError('Unknown option for bgremoval')

        print('Removing phase offset\n')
        raw_abs = np.abs(measclean)
        raw_ph = np.angle(measclean)
        clean_ph = -signal.detrend(np.unwrap(raw_ph - raw_ph[:, 0][:, None]))

        absclean = pd.DataFrame(raw_abs,
                                columns=measbgRe.pmtx.columns,
                                index=measbgRe.pmtx.index)
        phclean = pd.DataFrame(clean_ph,
                               columns=measbgRe.pmtx.columns,
                               index=measbgRe.pmtx.index)
        reclean = pd.DataFrame(np.real(raw_abs * np.exp(1j * clean_ph)),
                               columns=measbgRe.pmtx.columns,
                               index=measbgRe.pmtx.index)
        imclean = pd.DataFrame(np.imag(raw_abs * np.exp(1j * clean_ph)),
                               columns=measbgRe.pmtx.columns,
                               index=measbgRe.pmtx.index)

        print('Generating measmtx, measRe, measIm, measPh\n')
        self.measmtx = stlabutils.utils.stlabdict.stlabmtx(
            absclean,
            xtitle=measbgRe.pmtx.columns.name,
            ytitle=measbgRe.pmtx.index.name,
            ztitle='S11abs ()')
        self.measRe = stlabutils.utils.stlabdict.stlabmtx(
            reclean,
            xtitle=measbgRe.pmtx.columns.name,
            ytitle=measbgRe.pmtx.index.name,
            ztitle='S11re ()')
        self.measIm = stlabutils.utils.stlabdict.stlabmtx(
            imclean,
            xtitle=measbgRe.pmtx.columns.name,
            ytitle=measbgRe.pmtx.index.name,
            ztitle='S11im ()')
        self.measPh = stlabutils.utils.stlabdict.stlabmtx(
            phclean,
            xtitle=measbgRe.pmtx.columns.name,
            ytitle=measbgRe.pmtx.index.name,
            ztitle='S11ph (rad)')

        if dev == '2x2':
            print('\n# Loading in pkldump\n')
            pkldump = 'pkldump/2x2_processing_fit_RF_Vg_' + Vgstring + '_Pwrnobg.pkl'
            print(pkldump)
            mypkl = pickle.load(open(pkldump, 'rb'))
            print(mypkl.keys())

            print('\n# defining base parameters\n')
            self.Vg = mypkl['Vgate (V)'][0]
            self.pwrVNA = mypkl['Power (dBm)']
            self.f0 = mypkl['f0 (Hz)']
            self.f_0 = self.f0[0]
            self.kintfull = mypkl['kint (Hz)']
            self.kint = mypkl['kint (Hz)'][ikint]
            self.kext = mypkl['kext (Hz)'][0]
            self.k = self.kint + self.kext
        elif dev == '2x1':
            mydf = devdf

            self.Vg = mydf.index
            self.pwrVNA = measbgRe.pmtx.columns
            self.f0 = np.array([mydf['f0 (Hz)']] * len(self.pwrVNA))
            self.f_0 = self.f0[0]
            self.kintfull = np.array([mydf['kint (Hz)']] * len(self.pwrVNA))
            self.kint = mydf['kint (Hz)']
            self.kext = mydf['kext (Hz)']
            self.k = self.kint + self.kext
        else:
            raise ValueError('Unknown option for dev')

    def inspect_measurement_data(self, doplots=True, trimming=True):
        """
        Inspect measurement data and automatically crop around resonator
        """

        # get measurement data
        m_abs = self.measmtx.pmtx.values
        m_ph = self.measPh.pmtx.values
        meas_data = m_abs * np.exp(1j * m_ph)

        powers, freqs = self.measmtx.pmtx.axes

        # 2D meshgrids of measurement
        XX, YY = np.meshgrid(powers, freqs)
        DATA = meas_data.T

        if doplots:
            # plot measurement
            fig, ax = plt.subplots(constrained_layout=True)
            plt.axis('off')
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            plt.pcolormesh(XX, YY, np.abs(DATA))
            plt.xlabel('Power (dBm)')
            plt.ylabel('Frequency (Hz)')
            plt.title(r'$|S_{11}|$')
            ax2 = fig.add_subplot(gs[0, 1])
            plt.pcolormesh(XX, YY, np.angle(DATA))
            plt.xlabel('Power (dBm)')
            plt.ylabel('Frequency (Hz)')
            plt.title(r'$\angle S_{11}$')
            plt.show()
            plt.close()

        # trim measurement to area of interest
        if trimming == True:
            print('\n Trimming measurement to area of interest')
            myidx = np.logical_and(YY > self.f0[0] - 8 * self.k,
                                   YY < self.f0[0] + 4 * self.k)

            X2 = XX[myidx].reshape(len(YY[myidx]) // len(powers), len(powers))
            Y2 = YY[myidx].reshape(len(YY[myidx]) // len(powers), len(powers))
            DAT2 = DATA[myidx].reshape(
                len(YY[myidx]) // len(powers), len(powers))
        elif trimming == False:
            print('\n No data trimming')
            X2 = XX
            Y2 = YY
            DAT2 = DATA
        elif len(trimming) == 2:
            print('\n Trimming manually')
            myidx = np.logical_and(YY > trimming[0], YY < trimming[1])

            X2 = XX[myidx].reshape(len(YY[myidx]) // len(powers), len(powers))
            Y2 = YY[myidx].reshape(len(YY[myidx]) // len(powers), len(powers))
            DAT2 = DATA[myidx].reshape(
                len(YY[myidx]) // len(powers), len(powers))
        else:
            raise ValueError('unknown option for trimming')

        if doplots:
            # plot region of interest
            fig, ax = plt.subplots(constrained_layout=True)
            plt.axis('off')
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            plt.pcolormesh(X2, Y2, np.abs(DAT2))
            plt.xlabel('Power (dBm)')
            plt.ylabel('Frequency (Hz)')
            plt.title(r'$|S_{11}|$')
            ax2 = fig.add_subplot(gs[0, 1])
            plt.pcolormesh(X2, Y2, np.angle(DAT2))
            plt.xlabel('Power (dBm)')
            plt.ylabel('Frequency (Hz)')
            plt.title(r'$\angle S_{11}$')
            plt.suptitle('Measurement')
            plt.show()
            plt.close()

            fig, ax = plt.subplots(constrained_layout=True)
            plt.axis('off')
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            [plt.plot(yy, np.abs(dd)) for yy, dd in zip(Y2.T, DAT2.T)]
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(r'$|S_{11}|$')
            ax2 = fig.add_subplot(gs[0, 1])
            [plt.plot(yy, np.angle(dd)) for yy, dd in zip(Y2.T, DAT2.T)]
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(r'$\angle S_{11}$')
            plt.suptitle('meas All powers')
            plt.show()
            plt.close()

        self.X2, self.Y2, self.DAT2 = X2, Y2, DAT2
        return self.X2, self.Y2, self.DAT2

    def model_measurement_data(self,
                               att_av,
                               beta,
                               sol='min',
                               doplots=True,
                               kintval=None):
        """
        We will pass the model the inital cavity parameters, and the meshgrid of powers and frequencies 
        to evaluate the nonlinear cavity response at
        """

        if kintval is None:
            thekint = self.kint
        else:
            thekint = kintval
        S11theo = fitmodel(self.f0[0],
                           thekint,
                           self.kext,
                           beta,
                           att_av,
                           sol=sol,
                           P_out_VNA=self.X2.flatten(),
                           Probe_freq=self.Y2.flatten())

        # reshape simdata to do 2D plot
        S11theo2 = S11theo.reshape(self.DAT2.shape)

        if doplots:
            fig, ax = plt.subplots(constrained_layout=True)
            plt.axis('off')
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            plt.pcolormesh(self.X2, self.Y2, np.abs(S11theo2))
            plt.xlabel('Power (dBm)')
            plt.ylabel('Frequency (Hz)')
            plt.title('sim Abs(S11) ()')
            ax2 = fig.add_subplot(gs[0, 1])
            plt.pcolormesh(self.X2, self.Y2, np.angle(S11theo2))
            plt.xlabel('Power (dBm)')
            plt.ylabel('Frequency (Hz)')
            plt.title('sim Phase(S11) ()')
            plt.suptitle('Simulation')
            plt.show()
            plt.close()

            fig, ax = plt.subplots(constrained_layout=True)
            plt.axis('off')
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            [plt.plot(yy, np.abs(dd)) for yy, dd in zip(self.Y2.T, S11theo2.T)]
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Abs(S11) ()')
            ax2 = fig.add_subplot(gs[0, 1])
            [
                plt.plot(yy, np.angle(dd))
                for yy, dd in zip(self.Y2.T, S11theo2.T)
            ]
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase(S11) ()')
            plt.suptitle('sim All powers')
            plt.show()
            plt.close()

        self.S11theo2 = S11theo2
        return self.X2, self.Y2, self.S11theo2

    def fit_measurement_data(self,
                             att_av,
                             beta,
                             sol='min',
                             doplots=True,
                             betavary=True,
                             f0vary=False,
                             kintvary=False,
                             kextvary=False,
                             attavvary=False):
        """
        Pass the nonlinear model to lmfit to fit 2D data of S11 as a function of frequency and power.
        
        Parameters
        ----------
        att_av : float
            init attenuation, in dB
        beta : float
            init anharmonicity, in Hz
        sol : str, optional
            solution type, must be 'min' or 'max' or 'med'
        doplots : bool, optional
            make plots or not
        betavary: bool, optional
            whether to allow fit to vary beta
        f0vary: bool, optional
            whether to allow fit to vary f0
        kintvary: bool, optional
            whether to allow fit to vary kint
        kextvary: bool, optional
            whether to allow fit to vary kext
        attavvary: bool, optional
            whether to allow fit to vary att_av
        
        Returns
        -------
        result : lmfit ModelResult
            lmfit ModelResult of the S11 power dependence.
        fitdata : numpy 2D-array
            best fit result.
        """

        mymodel = lmfit.Model(fitmodel,
                              independent_vars=['P_out_VNA', 'Probe_freq'])
        print(mymodel.param_names, mymodel.independent_vars)

        params = mymodel.make_params()
        params.add('beta',
                   value=beta,
                   min=-2 * abs(beta),
                   max=-1 / 2 * abs(beta),
                   vary=betavary)
        params.add('f_0',
                   value=self.f0[0],
                   min=self.Y2.flatten().min(),
                   max=self.Y2.flatten().max(),
                   vary=f0vary)
        params.add('kint',
                   value=self.kint,
                   min=0.8 * np.nanmin(self.kintfull),
                   max=1.2 * np.nanmax(self.kintfull),
                   vary=kintvary)
        params.add('kext',
                   value=self.kext,
                   min=0.8 * self.kext,
                   max=1.2 * self.kext,
                   vary=kextvary)
        params.add('att_av',
                   value=att_av,
                   min=-1.2 * abs(att_av),
                   max=-0.8 * abs(att_av),
                   vary=attavvary)

        print(params.pretty_print())

        result = mymodel.fit(self.DAT2.flatten(),
                             P_out_VNA=self.X2.flatten(),
                             Probe_freq=self.Y2.flatten(),
                             params=params)

        print(result.fit_report())
        # reshape fitdata to do 2D plot
        fitdata = result.best_fit.reshape(self.DAT2.shape)
        self.fitdata = fitdata
        self.result = result

        return result, fitdata

    def probe_power_sim(self,
                        att_av,
                        beta,
                        P_out_VNA=np.linspace(-30, 10, 41),
                        sols=['min']):
        """
        DEPRECATED: simulate nonlinear resonator and dump sim results to file
        """

        f_0 = self.f_0
        k = self.k
        kext = self.kext
        Vg = self.Vg
        colnames = self.colnames

        print('# Probe power sim')
        # probe frequencies
        Probe_freqs = np.linspace(f_0 - 8 * k, f_0 + 3 * k, 201)
        self.Probe_freqs = Probe_freqs

        for sol in sols:

            print('Running Duffing sim for', sol, 'solution with beta =', beta,
                  'Hz')
            myfile = stlabutils.newfile(f"2x2_Vg{int(Vg):0d}_Is0",
                                        "ResponsePower_SingleTone_" + sol,
                                        mypath='simulations_Duffing/',
                                        usedate=False,
                                        usefolder=False,
                                        colnames=colnames,
                                        git_id=False)

            for i, _ in enumerate(P_out_VNA):

                for ii, f_p in enumerate(Probe_freqs):

                    # On-chip Probe Power
                    P_p_log = P_out_VNA[i] - abs(att_av)
                    if i == 0 and ii == 0:
                        print('using averaged input attenuation')
                    P_p_lin = 10**(P_p_log / 10) * 1e-3  # dBm to W

                    # On-chip photon flux
                    n_p = P_p_lin / hbar / 2 / pi / f_p

                    # Intracavity photon number
                    a = 4 * pi**2 * beta**2
                    b = -2 * 2 * pi * (f_p - f_0) * 2 * pi * beta
                    c = (2 * pi * (f_p - f_0))**2 + (2 * pi * k)**2 / 4
                    d = -2 * pi * kext * n_p
                    coeff = [a, b, c, d]
                    a_1, a_2, a_3 = np.roots(coeff)

                    # Replace complex roots by -1
                    roots = [a_1, a_2, a_3]
                    roots = [i.real if i.imag == 0 else -1 for i in roots]

                    # Choose branch:
                    # sol=='min': find smallest real and positive solution for the low-amplitude branch
                    # sol=='max': find largest real and positive solution for the high-amplitude branch
                    # sol=='med': find intermediate branch value
                    if sol == 'min':
                        alpha_0 = min([alpha for alpha in roots if alpha > 0])
                    elif sol == 'max':
                        alpha_0 = max([alpha for alpha in roots if alpha > 0])
                    elif sol == 'med':
                        alpha_0 = np.median(
                            [alpha for alpha in roots if alpha > 0])
                    # print(roots, alpha_0)

                    # Calculate response
                    Delta = f_p - f_0
                    phi = np.arctan(-2 * (Delta - beta * alpha_0) / k)
                    S11 = 1 - np.sqrt(2 * pi * kext * alpha_0) / np.sqrt(
                        n_p) * np.exp(-1j * phi)

                    line = [
                        f_p,
                        np.real(S11),
                        np.imag(S11), 20 * np.log10(np.abs(S11)),
                        np.angle(S11),
                        np.abs(S11), alpha_0, P_p_lin, P_p_log, beta,
                        P_out_VNA[i], Vg
                    ]
                    stlabutils.writeline(myfile, line)

                    # myfile.write('\n')
                myfile.write('\n')
                stlabutils.utils.metagen.fromarrays(
                    myfile,
                    Probe_freqs,
                    P_out_VNA[0:i + 1],
                    xtitle='Probe Frequency (Hz)',
                    ytitle='VNA power (dBm)',
                    colnames=colnames)

            self.fname = myfile.name
            myfile.close()

    def plot_data_sim(self, devpath, grid=True):
        """
        DEPRECATED: plot measurement and simulation data side by side
        """

        Vg = self.Vg
        measPh = self.measPh
        measmtx = self.measmtx
        measRe = self.measRe
        measIm = self.measIm
        Probe_freqs = self.Probe_freqs

        print('# Compare with data')
        print('Loading simdata\n')
        simdata = stlabutils.readdata.readdat(self.fname)
        simmtx = stlabutils.framearr_to_mtx(simdata,
                                            xkey='Probe Frequency (Hz)',
                                            ykey='VNA power (dBm)',
                                            key='S11abs ()')
        simRe = stlabutils.framearr_to_mtx(simdata,
                                           xkey='Probe Frequency (Hz)',
                                           ykey='VNA power (dBm)',
                                           key='S11re ()')
        simIm = stlabutils.framearr_to_mtx(simdata,
                                           xkey='Probe Frequency (Hz)',
                                           ykey='VNA power (dBm)',
                                           key='S11im ()')
        simPh = stlabutils.framearr_to_mtx(simdata,
                                           xkey='Probe Frequency (Hz)',
                                           ykey='VNA power (dBm)',
                                           key='S11ph (rad)')

        # plotting
        wbval = (0.1, 0.1)
        lims = np.percentile(simPh.pmtx.values, (wbval[0], 100 - wbval[1]))
        vmin = lims[0]
        vmax = lims[1]
        cmap = 'PiYG'

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.axis('off')
        gs = fig.add_gridspec(2, 2)
        plt.suptitle('Beta = {:0d} kHz, Vg= {:+0d} V'.format(
            int(simdata[0]['Beta (Hz)'][0] / 1e3), int(Vg)))

        ax00 = fig.add_subplot(gs[0, 0])
        plt.imshow(measPh.pmtx,
                   aspect='auto',
                   cmap=cmap,
                   extent=measmtx.getextents(),
                   vmin=vmin,
                   vmax=vmax)
        #plt.plot(fmin,pwrVNA,c='k')
        #plt.xlabel('Frequency (Hz)')
        plt.ylabel('VNA power (dBm)')
        plt.title('Phase (Data)')

        ax01 = fig.add_subplot(gs[0, 1])
        plt.imshow(simPh.pmtx,
                   aspect='auto',
                   cmap=cmap,
                   extent=simmtx.getextents(),
                   vmin=vmin,
                   vmax=vmax)
        #plt.plot(fmin,pwrVNA,c='k')
        #plt.xlabel('Frequency (Hz)')
        plt.ylabel('VNA power (dBm)')
        plt.title('Phase (Simulation)')

        cmap = 'PuBu_r'
        lims = np.percentile(simmtx.pmtx.values, (wbval[0], 100 - wbval[1]))
        vmin = lims[0]
        vmax = lims[1]

        ax10 = fig.add_subplot(gs[1, 0])
        plt.imshow(measmtx.pmtx,
                   aspect='auto',
                   cmap=cmap,
                   extent=measmtx.getextents(),
                   vmin=vmin,
                   vmax=vmax)
        #plt.plot(fmin,pwrVNA,c='k')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('VNA power (dBm)')
        plt.title('Abs (Data)')

        ax11 = fig.add_subplot(gs[1, 1])
        plt.imshow(simmtx.pmtx,
                   aspect='auto',
                   cmap=cmap,
                   extent=simmtx.getextents(),
                   vmin=vmin,
                   vmax=vmax)
        #plt.plot(fmin,pwrVNA,c='k')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('VNA power (dBm)')
        plt.title('Abs (Simulation)')

        for theax in [ax00, ax01, ax10, ax11]:
            plt.sca(theax)
            plt.xlim(min(Probe_freqs), max(Probe_freqs))
            if grid:
                plt.grid()

        gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
        filename = 'plots/' + devpath + f'modeling_vs_Ppump_2D_{int(Vg):+0d}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 8))

        plt.axis('off')
        gs = fig.add_gridspec(4, 4)
        plt.suptitle('Beta = {:0d} kHz, Vg= {:+0d} V'.format(
            int(simdata[0]['Beta (Hz)'][0] / 1e3), int(Vg)))

        idx = 0
        ax0imre = fig.add_subplot(gs[0:2, 0])
        plt.plot(measRe.pmtx.iloc[idx],
                 measIm.pmtx.iloc[idx],
                 '.-',
                 label=measmtx.pmtx.axes[0][idx])
        plt.plot(simRe.pmtx.iloc[idx],
                 simIm.pmtx.iloc[idx],
                 label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlabel('Real S11 ()')
        plt.ylabel('Imag S11 ()')

        ax0abs = fig.add_subplot(gs[0, 1])
        plt.plot(measmtx.pmtx.iloc[idx], '.-', label=measmtx.pmtx.axes[0][idx])
        plt.plot(simmtx.pmtx.iloc[idx], label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlim(min(Probe_freqs), max(Probe_freqs))
        plt.ylabel('Abs S11 ()')

        ax0ph = fig.add_subplot(gs[1, 1])
        plt.plot(measPh.pmtx.iloc[idx], '.-', label=measmtx.pmtx.axes[0][idx])
        plt.plot(simPh.pmtx.iloc[idx], label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlim(min(Probe_freqs), max(Probe_freqs))
        plt.ylabel('Phase S11 (rad)')

        idx = 20
        ax1imre = fig.add_subplot(gs[:2, 2])
        plt.plot(measRe.pmtx.iloc[idx],
                 measIm.pmtx.iloc[idx],
                 '.-',
                 label=measmtx.pmtx.axes[0][idx])
        plt.plot(simRe.pmtx.iloc[idx],
                 simIm.pmtx.iloc[idx],
                 label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlabel('Real S11 ()')
        plt.ylabel('Imag S11 ()')

        ax1abs = fig.add_subplot(gs[0, 3])
        plt.plot(measmtx.pmtx.iloc[idx], '.-', label=measmtx.pmtx.axes[0][idx])
        plt.plot(simmtx.pmtx.iloc[idx], label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlim(min(Probe_freqs), max(Probe_freqs))
        plt.ylabel('Abs S11 ()')

        ax1ph = fig.add_subplot(gs[1, 3])
        plt.plot(measPh.pmtx.iloc[idx], '.-', label=measmtx.pmtx.axes[0][idx])
        plt.plot(simPh.pmtx.iloc[idx], label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlim(min(Probe_freqs), max(Probe_freqs))
        plt.ylabel('Phase S11 (rad)')

        idx = 30
        ax2imre = fig.add_subplot(gs[2:, 0])
        plt.plot(measRe.pmtx.iloc[idx],
                 measIm.pmtx.iloc[idx],
                 '.-',
                 label=measmtx.pmtx.axes[0][idx])
        plt.plot(simRe.pmtx.iloc[idx],
                 simIm.pmtx.iloc[idx],
                 label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlabel('Real S11 ()')
        plt.ylabel('Imag S11 ()')

        ax2abs = fig.add_subplot(gs[2, 1])
        plt.plot(measmtx.pmtx.iloc[idx], '.-', label=measmtx.pmtx.axes[0][idx])
        plt.plot(simmtx.pmtx.iloc[idx], label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlim(min(Probe_freqs), max(Probe_freqs))
        plt.ylabel('Abs S11 ()')

        ax2ph = fig.add_subplot(gs[3, 1])
        plt.plot(measPh.pmtx.iloc[idx], '.-', label=measmtx.pmtx.axes[0][idx])
        plt.plot(simPh.pmtx.iloc[idx], label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlim(min(Probe_freqs), max(Probe_freqs))
        plt.ylabel('Phase S11 (rad)')

        idx = 40
        ax3imre = fig.add_subplot(gs[2:, 2])
        plt.plot(measRe.pmtx.iloc[idx],
                 measIm.pmtx.iloc[idx],
                 '.-',
                 label=measmtx.pmtx.axes[0][idx])
        plt.plot(simRe.pmtx.iloc[idx],
                 simIm.pmtx.iloc[idx],
                 label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlabel('Real S11 ()')
        plt.ylabel('Imag S11 ()')

        ax3abs = fig.add_subplot(gs[2, 3])
        plt.plot(measmtx.pmtx.iloc[idx], '.-', label=measmtx.pmtx.axes[0][idx])
        plt.plot(simmtx.pmtx.iloc[idx], label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlim(min(Probe_freqs), max(Probe_freqs))
        plt.ylabel('Abs S11 ()')

        ax3ph = fig.add_subplot(gs[3, 3])
        plt.plot(measPh.pmtx.iloc[idx], '.-', label=measmtx.pmtx.axes[0][idx])
        plt.plot(simPh.pmtx.iloc[idx], label=measmtx.pmtx.axes[0][idx])
        plt.legend()
        plt.xlim(min(Probe_freqs), max(Probe_freqs))
        plt.ylabel('Phase S11 (rad)')

        gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
        filename = 'plots/' + devpath + f'modeling_vs_Ppump_{int(Vg):+0d}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()
