# -*- coding: utf-8 -*-
#import corner
import lmfit
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import pandas as pd
import pickle
from src.model import f0 as f0model
from src.model import Lj as Ljmodel
from src.model import Icmodel, f0_of_I, Lj_of_I


def resid(params, x, ydata):
    fr = params['fr'].value
    Lr = params['Lr'].value
    Ic = params['Ic'].value
    tau = params['tau'].value

    y_model = f0_of_I(fr, Lr, x, Ic, tau)
    return y_model - ydata


class DataBestTau:
    """
    Generate and fit model data using nonsinusoidal CPRs in an attempt to estimate the best value of tau via minimizing the reduced chi-square value
    """
    def __init__(self, **kwargs):

        for key, val in kwargs.items():
            setattr(self, key, val)

    def find_best_tau(self, verbose=True, logscale=True):
        """
        For the input dict `resdata` and the array `mytaus`, fit the input data using a fix value of tau.
        We then calculate the reduced chi-square for each of these fits with a different value of tau_fix.
        We plot the reduced chi-square for the mytaus and indicate the tau with lowest chi-square.
        This tau_fix we store in self.besttau
        """

        resdata, mytaus = self.resdata, self.mytaus
        redchi = []
        newtaus = np.linspace(mytaus.min(), mytaus.max(), 101)

        for tau in newtaus:

            mymodel = lmfit.Model(f0_of_I, independent_vars=['I'])
            params = lmfit.Parameters()
            params.add('fr', value=resdata['fr'])
            params.add('Lr', value=resdata['Lr'])
            params.add('Ic', value=resdata['Ic'], min=max(resdata['x']))
            params.add('tau', value=tau, vary=False, min=0, max=1)

            #o1 = lmfit.minimize(resid, params, args=(resdata['x'], resdata['y']))
            o1 = mymodel.fit(resdata['y'],
                             params,
                             I=resdata['x'],
                             weights=resdata['weights'])
            redchi.append(o1)

        qty = np.array([x.redchi for x in redchi])
        idx = np.argmin(qty)
        if verbose:
            print(redchi[idx].fit_report())

            plt.plot(newtaus, qty)
            plt.plot(newtaus[idx], qty[idx], 'o')
            if logscale:
                plt.yscale('log')
            plt.show()
            plt.close()

            print(newtaus[idx], qty[idx])

        self.bestfr, self.bestLr, self.bestIc = redchi[idx].params[
            'fr'].value, redchi[idx].params['Lr'].value, redchi[idx].params[
                'Ic'].value
        self.besttau = newtaus[idx]

        return redchi

    def fit_best_tau(self, verbose=True):
        """
        Using the self.besttau extracted in self.find_best_tau(), we fit the original data with this tau as starting value,
        now fitting all four parameters at once.
        Returns the fitresult
        """

        resdata = self.resdata

        mymodel = lmfit.Model(f0_of_I, independent_vars=['I'])
        params = lmfit.Parameters()
        params.add('fr', value=self.bestfr)
        params.add('Lr', value=self.bestLr)
        params.add('Ic', value=self.bestIc)
        params.add('tau', value=self.besttau, min=0, max=1)

        #o1 = lmfit.minimize(resid, params, args=(resdata['x'], resdata['y']))
        o1 = mymodel.fit(resdata['y'],
                         params,
                         I=resdata['x'],
                         weights=resdata['weights'])
        if verbose:
            print(o1.fit_report())

        return o1


class ResidualOverview:
    """
    Generate and fit model data using nonsinusoidal CPRs
    """
    def __init__(self, **kwargs):

        for key, val in kwargs.items():
            setattr(self, key, val)

    def residual_fun(self, tau_vary=False, tau_0=0, verbose=True, data=False):

        fr = self.fr
        Lr = self.Lr
        Ic = self.Ic
        mytaus = self.mytaus
        if hasattr(self, 'mycurr'):
            mycurr = self.mycurr
            x = mycurr * Ic
        else:
            x = data['x']
            self.mycurr = x

    def curr_tau_map(self, tau_vary=False, tau_0=0, verbose=True, data=False):

        fr = self.fr
        Lr = self.Lr
        Ic = self.Ic
        mytaus = self.mytaus
        if hasattr(self, 'mycurr'):
            mycurr = self.mycurr
            x = mycurr * Ic
        else:
            x = data['x']
            self.mycurr = x

        f0results = []
        self.testdata = []

        for mytau in mytaus:
            if verbose:
                print('\n########### mytau:', mytau)
            if not data:
                y = f0_of_I(fr, Lr, x, Ic, tau=mytau)
            else:
                y = data['y']
                weights = data['weights']
            self.testdata.append(y)

            mymodel = lmfit.Model(f0_of_I, independent_vars=['I'])
            #mymodel.set_param_hint('fr', min=np.max(y),max=2*np.max(y))
            #mymodel.set_param_hint('Lr', min=0, max=1)
            mymodel.set_param_hint('Ic', min=max(x))
            if tau_vary:
                mymodel.set_param_hint('tau', min=0, max=1)
            else:
                mymodel.set_param_hint('tau', vary=False)
            if tau_0 == 'sweep':
                f0params = mymodel.make_params(fr=fr, Lr=Lr, Ic=Ic, tau=mytau)
            else:
                f0params = mymodel.make_params(fr=fr, Lr=Lr, Ic=Ic, tau=tau_0)

            if not data:
                f0result = mymodel.fit(y, f0params, I=x)
            else:
                f0result = mymodel.fit(y, f0params, I=x, weights=weights)
            f0results.append(f0result)
            """f0result.plot()
            plt.show()
            plt.close()"""

            if verbose:
                print(f0result.fit_report())

        mydf = pd.DataFrame({
            'Lj (H)': [Lj_of_I(0, Ic, tau=x) for x in mytaus],
            'fr (Hz)': [x.params['fr'].value for x in f0results],
            'Lr (H)': [x.params['Lr'].value for x in f0results],
            'Ic (A)': [x.params['Ic'].value for x in f0results],
            'tau_fit': [x.params['tau'].value for x in f0results],
            'dfr (Hz)': [x.params['fr'].stderr for x in f0results],
            'dLr (H)': [x.params['Lr'].stderr for x in f0results],
            'dIc (A)': [x.params['Ic'].stderr for x in f0results],
            'dtau_fit': [x.params['tau'].stderr for x in f0results],
            'fr0 (Hz)':
            fr,
            'Lr0 (H)':
            Lr,
            'Ic0 (A)':
            Ic,
            'tau0':
            mytaus,
            'chisqr': [x.chisqr for x in f0results],
            'redchi': [x.redchi for x in f0results]
        })

        self.mydf = mydf
        self.mykeys = mydf.keys()
        self.myparams = f0results[-1].params
        self.residuals = [x.residual for x in f0results]

        return f0results

    def plot_fitresults(self, savefig=False, errors=True, **kwargs):

        mydf = self.mydf
        mykeys = self.mykeys
        myparams = self.myparams
        parlen = len(myparams)

        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        gs = fig.add_gridspec(1, parlen + 1)
        k = 1
        i = 0

        # to return
        xs, ys, yerrs, vals, ytrues = [], [], [], [], []

        for j in range(parlen + 1):
            val = mykeys[j]
            theax = fig.add_subplot(gs[0, j])
            x = mydf['tau0'].values
            y = mydf[val].values

            xs.append(x)
            ys.append(y)
            vals.append(val)

            if j != 0:
                yerr = mydf[mykeys[j + parlen]].values
                ytrue = mydf[mykeys[j + 2 * parlen]].values
                plt.plot(x, y, 'none')
                plt.plot(x, ytrue, c='C3', ls='--', zorder=99)
                ytrues.append(ytrue)
            else:
                plt.plot(x, y, '-', label=val)
                yerrs.append([0]*len(y))
            #plt.yscale('log')
            mylims = plt.gca().get_ylim()
            if j != 0:
                if errors:
                    yerr = [x if x is not None else 0 for x in yerr]
                    plt.errorbar(x, y, yerr, fmt='-', label=val)
                    yerrs.append(yerr)

                else:
                    plt.plot(x, y, label=val)
                    yerrs.append([0] * len(y))
            plt.ylim(mylims)
            plt.xlabel('Transparency (norm)')
            plt.ylabel(val)
            plt.legend()
            k += 1

        plt.suptitle('Bias model fit for nonsinusoidal CPRs')
        gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
        if savefig:
            filename = 'plots/modeling_CPR_f0_vs_Ib.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()

        resdf = pd.DataFrame({'tau': xs[0]})
        resdf = resdf.set_index('tau')
        for i, (val, y, yerr) in enumerate(zip(vals, ys, yerrs)):
            resdf[val] = y
            resdf['d' + val] = yerr
        resdf['chisqr']=mydf['chisqr'].values
        resdf['redchi']=mydf['redchi'].values

        return resdf

    def plot_residual2D(self,
                        savefig=False,
                        cmap='binary_r',
                        note='',
                        **kwargs):

        residuals = np.array(self.residuals)
        testdata = np.array(self.testdata)
        Xres, Yres = np.meshgrid(self.mycurr, self.mytaus)
        mydf = self.mydf

        fig = plt.figure(figsize=(12, 4))  #,constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 3])
        ax1 = fig.add_subplot(gs[0, 0])
        plt.plot(mydf['tau0'], np.sqrt(mydf['chisqr']) / 1e3)
        plt.ylabel('$\sqrt{\chi^2}$ (kHz)')
        ax2 = fig.add_subplot(gs[0, 1])
        plt.plot(mydf['tau0'], np.sqrt(mydf['redchi']) / 1e3)
        plt.ylabel('reduced $\sqrt{\chi^{2}}$ (kHz)')
        for theax in [ax1, ax2]:
            plt.sca(theax)
            #plt.yscale('log')
            plt.xlabel('Transparency (norm)')

        ax3 = fig.add_subplot(gs[0, 2])
        wbval = (0.1, 0.1)
        lims = np.percentile(abs(residuals.flatten()),
                             (wbval[0], 100 - wbval[1]))
        vmin = lims[0]
        vmax = lims[1]
        #vmin = -residuals.flatten().min()
        #vmax = residuals.flatten().min()

        plt.pcolormesh(Xres,
                       Yres,
                       abs(residuals) / 1e3,
                       vmin=vmin / 1e3,
                       vmax=vmax / 1e3,
                       norm=colors.LogNorm(),
                       cmap=cmap)
        cbar = plt.colorbar()
        plt.xlabel('Bias current (µA)')
        plt.ylabel('Transparency (norm)')
        cbar.ax.set_ylabel('Residuals (kHz)')

        plt.suptitle('Bias model fit for nonsinusoidal CPRs')
        #plt.tight_layout()
        gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
        if savefig:
            filename = 'plots/modeling_CPR_f0_vs_Ib_residuals' + note + '.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()

        return residuals, testdata, Xres, Yres, vmin, vmax


class ResidualWrapper:
    """
    Calculate fit residuals and chi-squared values for varying fit parameters
    """
    def __init__(self, **kwargs):

        self.imax = 0.9
        for key, val in kwargs.items():
            setattr(self, key, val)

    def make_grid(self):

        x = self.Ic * np.linspace(0, self.imax, 101)
        y = f0_of_I(self.fr, self.Lr, x, self.Ic, tau=self.mytau)

        self.Xfr, self.YLr = np.meshgrid(self.test_fr, self.test_Lr)
        diffs1 = []
        for xfr, ylr in zip(self.Xfr.flatten(), self.YLr.flatten()):
            ynew = f0_of_I(xfr, ylr, x, self.Ic, tau=self.mytau)
            diff = sum((y - ynew)**2)
            diffs1.append(diff)
        self.diffs1 = np.array(diffs1).reshape(self.Xfr.shape)

        self.XIc, self.YLr = np.meshgrid(self.test_Ic, self.test_Lr)
        diffs2 = []
        for xIc, yLr in zip(self.XIc.flatten(), self.YLr.flatten()):
            ynew = f0_of_I(self.fr, yLr, x, xIc, tau=self.mytau)
            diff = sum((y - ynew)**2)
            diffs2.append(diff)
        self.diffs2 = np.array(diffs2).reshape(self.Xfr.shape)

        self.Xfr, self.YIc = np.meshgrid(self.test_fr, self.test_Ic)
        diffs3 = []
        for xfr, yIc in zip(self.Xfr.flatten(), self.YIc.flatten()):
            ynew = f0_of_I(xfr, self.Lr, x, yIc, tau=self.mytau)
            diff = sum((y - ynew)**2)
            diffs3.append(diff)
        self.diffs3 = np.array(diffs3).reshape(self.Xfr.shape)

        return self.diffs1, self.diffs2, self.diffs3

    def plot(self, savefig=False, contour=False, contourf=False, **kwargs):

        wbval = (0.1, 0.1)
        vmin1, vmax1 = np.percentile(self.diffs1.flatten(),
                                     (wbval[0], 100 - wbval[1]))
        vmin2, vmax2 = np.percentile(self.diffs2.flatten(),
                                     (wbval[0], 100 - wbval[1]))
        vmin3, vmax3 = np.percentile(self.diffs3.flatten(),
                                     (wbval[0], 100 - wbval[1]))

        fig = plt.figure(figsize=(15, 4))
        gs = fig.add_gridspec(1, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        if contourf:
            CS = plt.contourf(self.Xfr / 1e9,
                              self.YLr / 1e-9,
                              self.diffs1,
                              vmin=vmin1,
                              vmax=vmax1,
                              norm=colors.LogNorm(),
                              **kwargs)
        else:
            CS = plt.pcolormesh(self.Xfr / 1e9,
                                self.YLr / 1e-9,
                                self.diffs1,
                                vmin=vmin1,
                                vmax=vmax1,
                                norm=colors.LogNorm(),
                                **kwargs)
        cbar = plt.colorbar(CS)
        CS.set_edgecolor('face')
        if contour:
            plt.contour(self.Xfr / 1e9,
                        self.YLr / 1e-9,
                        self.diffs1,
                        vmin=vmin1,
                        vmax=vmax1,
                        norm=colors.LogNorm(),
                        colors='k')
        plt.xlabel('fr (GHz)')
        plt.ylabel('Lr (nH)')
        cbar.ax.set_ylabel('$\chi^2$ (Hz$^2$)')

        ax2 = fig.add_subplot(gs[0, 1])
        plt.pcolormesh(self.YLr / 1e-9,
                       self.XIc / 1e-6,
                       self.diffs2,
                       vmin=vmin2,
                       vmax=vmax2,
                       norm=colors.LogNorm(),
                       **kwargs)
        cbar = plt.colorbar()
        plt.xlabel('Lr (nH)')
        plt.ylabel('Ic (µA)')
        cbar.ax.set_ylabel('$\chi^2$ (Hz$^2$)')

        ax3 = fig.add_subplot(gs[0, 2])
        plt.pcolormesh(self.Xfr / 1e9,
                       self.YIc / 1e-6,
                       self.diffs3,
                       vmin=vmin3,
                       vmax=vmax3,
                       norm=colors.LogNorm(),
                       **kwargs)
        cbar = plt.colorbar()
        plt.xlabel('fr (GHz)')
        plt.ylabel('Ic (µA)')
        cbar.ax.set_ylabel('$\chi^2$ (Hz$^2$)')

        plt.suptitle(
            r'Residuals for bias model of nonsinusoidal CPR, $\tau={:.2f}$, $i_m={:.2f}$ $I_c$'
            .format(self.mytau, self.imax))
        gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
        if savefig:
            filename = f'plots/residuals_CPR_f0_vs_Ib_testdata_{self.mytau:.2f}.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()


class EmceeWrapper:
    """
    Calculate fit covariance, correlations and uncertainties using emcee.
    Literature:
    - https://lmfit.github.io/lmfit-py/examples/example_emcee_Model_interface.html
    - https://lmfit.github.io/lmfit-py/examples/documentation/fitting_emcee.html#sphx-glr-examples-documentation-fitting-emcee-py
    - https://lmfit.github.io/lmfit-py/examples/documentation/confidence_advanced.html#sphx-glr-examples-documentation-confidence-advanced-py
    - https://scraps.readthedocs.io/en/latest/Example1_LoadAndPlot.html
    - https://scraps.readthedocs.io/en/latest/Example3_FiguresForManuscript.html
    - https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee
    """
    def __init__(self, tau_vary=False, **kwargs):

        self.tau_0 = 0
        for key, val in kwargs.items():
            setattr(self, key, val)

        fr, Lr, Ic, tau = self.fr, self.Lr, self.Ic, self.mytau
        mycurr = self.mycurr

        self.model = lmfit.Model(f0_of_I, independent_vars=['I'])

        truths = (fr, Lr, Ic, tau)
        x = Ic * mycurr
        np.random.seed(0)
        y = f0_of_I(fr, Lr, x, Ic, tau=tau)

        self.p = self.model.make_params(fr=fr, Lr=Lr, Ic=Ic, tau=self.tau_0)
        if tau_vary:
            self.p['tau'].min = 0
        else:
            self.p['tau'].vary = False
        """
            if tau_0 == 'sweep':
                f0params = mymodel.make_params(fr=fr, Lr=Lr, Ic=Ic, tau=mytau)
            else:
                f0params = mymodel.make_params(fr=fr, Lr=Lr, Ic=Ic, tau=tau_0)"""

        self.result = self.model.fit(
            data=y, params=self.p, I=x)  #, method='Nelder', nan_policy='omit')

        print(lmfit.report_fit(self.result))
        self.result.plot()
        plt.show()
        plt.close()

        self.x, self.y = x, y

    def execute(self, steps=10000, burn=300, thin=20, nwalkers=200):

        self.emcee_kws = dict(steps=steps,
                              burn=burn,
                              thin=thin,
                              nwalkers=nwalkers,
                              is_weighted=False,
                              progress=False)
        self.emcee_params = self.result.params.copy()
        self.emcee_params.add('__lnsigma',
                              value=10)  #, min=np.log(0.001), max=np.log(2.0))

        print('\nemcee_params:')
        print(self.emcee_params.pretty_print())

        self.result_emcee = self.model.fit(data=self.y,
                                           I=self.x,
                                           params=self.emcee_params,
                                           method='emcee',
                                           nan_policy='omit',
                                           fit_kws=self.emcee_kws)

        print(lmfit.report_fit(self.result_emcee))

    def execute_plot(self, **kwargs):

        ax = plt.plot(self.x,
                      self.model.eval(params=self.result.params, I=self.x),
                      label='Nelder',
                      zorder=100)
        self.result_emcee.plot_fit(ax=ax,
                                   data_kws=dict(color='gray', markersize=2))
        plt.show()
        plt.close()

    def check_walkers(self, **kwargs):

        plt.plot(self.result_emcee.acceptance_fraction)
        plt.xlabel('walker')
        plt.ylabel('acceptance fraction')
        plt.show()
        plt.close()

        if hasattr(self.result_emcee, "acor"):
            print("Autocorrelation time for the parameters:")
            print("----------------------------------------")
            i = 0
            for p in self.result_emcee.params:
                if p != 'nJJ':
                    print(p, self.result_emcee.acor[i])
                    i += 1

    def cornerplot(self, savefig=True, **kwargs):

        emcee_corner = corner.corner(
            self.result_emcee.flatchain,
            labels=self.result_emcee.var_names,
            truths=list(self.result_emcee.params.valuesdict().values()))
        plt.suptitle(
            r'Corner plot for bias model of nonsinusoidal CPR, $\tau={:.2f}$, $i_m={:.2f}$ $I_c$'
            .format(self.mytau,
                    self.x.max() / self.Ic))
        if savefig:
            filename = f'plots/corner_CPR_f0_vs_Ib_testdata_{self.mytau:.2f}.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()

    def error_estimates(self, **kwargs):

        result_emcee = self.result_emcee
        emcee_params = self.emcee_params

        print("\nmedian of posterior probability distribution")
        print('--------------------------------------------')
        print(lmfit.report_fit(result_emcee.params))

        # find the maximum likelihood solution
        highest_prob = np.argmax(result_emcee.lnprob)
        hp_loc = np.unravel_index(highest_prob, result_emcee.lnprob.shape)
        mle_soln = result_emcee.chain[hp_loc]
        print("\nMaximum likelihood Estimation")
        print('-----------------------------')
        ix = 0
        for param in emcee_params:
            if param != 'nJJ':
                print(param + ': ' + str(mle_soln[ix]))
                ix += 1

        quantiles = np.percentile(result_emcee.flatchain['Ic'],
                                  [2.28, 15.9, 50, 84.2, 97.7])
        print("\n\n1 sigma spread", 0.5 * (quantiles[3] - quantiles[1]))
        print("2 sigma spread", 0.5 * (quantiles[4] - quantiles[0]))


class DataWrapper:
    """
    Compare nonsinusoidal CPR to measured data
    """
    def __init__(self, **kwargs):

        for key, val in kwargs.items():
            setattr(self, key, val)

        foundVg = False
        for i, myfile in enumerate(self.myfiles):
            vgstr = myfile.split('_')[-2]
            theVg = float(vgstr)
            if theVg == self.Vg:
                print("Processing " + vgstr)
                foundVg = True
                self.myfile = myfile
                continue
        if not foundVg:
            print('Could not find file with Vg=', self.Vg,
                  ',try another one out of these:')
            print(self.myfiles)
            raise ValueError

    def get_fit_ready(self, modelfile, mytaus, **kwargs):

        # Cavity parameters assuming varying values (further down indicated with "_fit")
        fitpars = pickle.load(open(self.myfile, "rb"))
        f0fit = fitpars['f0 (Hz)']
        Iset = fitpars['Iset (A)']
        df0fit = fitpars['df0 (Hz)']
        self.Vg = np.nanmax(fitpars['Vgate (V)']) # overwriting Vg

        modeling_all = pickle.load(open(modelfile, 'rb'))
        whichloc = int(
            np.argwhere(modeling_all['Vgate (V)'].values == self.Vg))
        myslice = modeling_all.iloc[whichloc]
        fr_data = myslice['fr (Hz)']
        Lr_data = myslice['Lr (H)']
        Ic_data = myslice['Ic (A)']

        xdata = Iset[~np.isnan(f0fit)]
        ydata = f0fit[~np.isnan(f0fit)]
        err = df0fit[~np.isnan(f0fit)]
        weights = 1. / err
        nanidx = np.where(np.isnan(weights))[0]
        weights[nanidx] = min(weights)  # if there is still a nan value left

        # drop last resonance frequency if it is higher than the second last
        if ydata[-1] > ydata[-2]:
            xdata = xdata[:-1]
            ydata = ydata[:-1]
            weights = weights[:-1]

        return {
            'x': xdata,
            'y': ydata,
            'weights': weights,
            'fr': fr_data,
            'Lr': Lr_data,
            'Ic': Ic_data
        }, ResidualOverview(Ic=Ic_data, fr=fr_data, Lr=Lr_data, mytaus=mytaus)
