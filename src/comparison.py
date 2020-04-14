# -*- coding: utf-8 -*-
from src.model import f0 as f0model
from src.model import Lj as Ljmodel
from src.model import Icmodel
import matplotlib.pyplot as plt
import numpy as np
import os


class Comparison:
    """
    Class for comparing DC and RF currents and inductances
    """
    def __init__(self, Ib=True, **kwargs):

        self.averopt = 5

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.Ib = Ib
        self.Vcnp = self.rfpkl_Vg['Vcnp (V)']
        if self.Ib:
            self.fitvg = self.fitpkl_Ib['Vgate (V)']
            if self.averopt == 'all':
                indices = ~(0 == self.fitpkl_Ib['dfr (Hz)'])
            else:
                indices = (self.fitvg > self.averopt)
            if not hasattr(self, 'frave'):
                # allows for passing a pre-calculated average to __init__
                self.frave = np.average(self.fitpkl_Ib['fr (Hz)'][indices],
                                        weights=1 /
                                        self.fitpkl_Ib['dfr (Hz)'][indices])
                self.dfrave = np.sqrt(1/np.sum(1 /
                                        self.fitpkl_Ib['dfr (Hz)'][indices]**2))
            if not hasattr(self, 'Lrave'):
                # allows for passing a pre-calculated average to __init__
                self.Lrave = np.average(self.fitpkl_Ib['Lr (H)'][indices],
                                        weights=1 /
                                        self.fitpkl_Ib['dLr (H)'][indices])
                self.dLrave = np.sqrt(1/np.sum(1 /
                                        self.fitpkl_Ib['dLr (H)'][indices]**2))

            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            plt.plot(self.fitvg, self.fitpkl_Ib['fr (Hz)'], c='tab:blue')
            mylims = ax1.get_ylim()
            plt.errorbar(self.fitvg,
                         self.fitpkl_Ib['fr (Hz)'],
                         self.fitpkl_Ib['dfr (Hz)'],
                         fmt='.-',
                         c='tab:blue')
            plt.axhline(self.frave, c='tab:red')
            plt.ylim(mylims)
            plt.xlabel('Vgate (V)')
            plt.ylabel('fr (Hz)')

            ax2 = fig.add_subplot(gs[0, 1])
            plt.plot(self.fitvg, self.fitpkl_Ib['Lr (H)'])
            plt.yscale('log')
            mylims = ax2.get_ylim()
            plt.errorbar(self.fitvg,
                         self.fitpkl_Ib['Lr (H)'],
                         self.fitpkl_Ib['dLr (H)'],
                         fmt='.-',
                         c='tab:blue')
            plt.axhline(self.Lrave, c='tab:red')
            plt.ylim(mylims)
            plt.xlabel('Vgate (V)')
            plt.ylabel('Lr (H)')

            plt.show()
            plt.close()
        else:
            print(
                'Warning: no Ib pkl! might not be able to plot everything...')

    def compare_Ic_Vg(self, logscale=True):

        plt.title('Ic vs Vgate')
        plt.plot(self.dcpkl_Vg["Vgate (V)"],
                 self.dcpkl_Vg["Iswitch (A)"],
                 '.',
                 label='DC data')
        if logscale:
            plt.yscale('log')
        mylims = plt.gca().get_ylim()
        if self.Ib:
            plt.errorbar(self.fitvg,
                         self.fitpkl_Ib['Ic (A)'],
                         self.fitpkl_Ib['dIc (A)'],
                         fmt='.-',
                         label='RF fit')
            plt.plot(self.dcpkl_Vg["Vgate (V)"][::-1],
                     Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'],
                             fr=self.frave,
                             Lr=self.Lrave),
                     label='RF fit ave')
        plt.xlabel('Vgate (V)')
        plt.ylabel('Ic (A)')
        plt.ylim(mylims)
        plt.legend(loc='best')
        plt.show()
        plt.close()

    def compare_f0_Vg(self, DC=True):
        plt.title('f0 vs Vgate')
        plt.errorbar(self.rfpkl_Vg["Vgate (V)"],
                     self.rfpkl_Vg["f0 (Hz)"],
                     self.rfpkl_Vg["df0 (Hz)"],
                     fmt='.',
                     label='RF data',
                     zorder=-99)
        if self.Ib:
            plt.plot(self.fitvg,
                     f0model(I=0,
                             Ic=self.fitpkl_Ib['Ic (A)'],
                             fr=self.fitpkl_Ib['fr (Hz)'],
                             Lr=self.fitpkl_Ib['Lr (H)']),
                     '.-',
                     label='RF fit')
            plt.plot(self.fitvg,
                     f0model(I=0,
                             Ic=self.fitpkl_Ib['Ic (A)'],
                             fr=self.frave,
                             Lr=self.Lrave),
                     '.-',
                     label='RF fit ave')
        if DC:
            plt.plot(self.dcpkl_Vg["Vgate (V)"],
                     f0model(I=0,
                             Ic=self.dcpkl_Vg["Iswitch (A)"],
                             fr=self.fitpkl_Vg['fr'],
                             Lr=self.fitpkl_Vg['Lr']),
                     label='DC extrapolation')
        plt.xlabel('Vgate (V)')
        plt.ylabel('f0 (Hz)')
        plt.legend(loc='best')
        plt.show()
        plt.close()

    def compare_f0_Ic(self, logscale=True):

        plt.title('f0 vs Ic')
        plt.errorbar(self.dcpkl_Vg["Iswitch (A)"][::-1],
                     self.rfpkl_Vg["f0 (Hz)"],
                     self.rfpkl_Vg["df0 (Hz)"],
                     fmt='.',
                     label='RF data',
                     zorder=-99)
        plt.plot(self.dcpkl_Vg["Iswitch (A)"],
                 f0model(I=0,
                         Ic=self.dcpkl_Vg["Iswitch (A)"],
                         fr=self.fitpkl_Vg['fr'],
                         Lr=self.fitpkl_Vg['Lr']),
                 label='RF fit')
        #plt.plot(dcpkl_Vg["Iswitch (A)"][::-1],f0model(I=0,Ic=fitpkl_Vg['Icrf'],fr=fitpkl_Vg['fr'],Lr=fitpkl_Vg['Lr'])
        #             ,label='Ic as fit parameter for RF fit')
        if logscale:
            plt.xscale('log')
        #plt.xlim(1e-7,1e-5)
        plt.xlabel('Iswitch (A)')
        plt.ylabel('f0 (Hz)')
        plt.legend(loc='best')
        plt.show()
        plt.close()

    def compare_Ic_Ic(self, extension='', save=True, splitcnp=True):

        id1 = self.dcpkl_Vg['Vgate (V)'] <= self.Vcnp
        id2 = self.dcpkl_Vg['Vgate (V)'] > self.Vcnp
        x1 = self.dcpkl_Vg["Iswitch (A)"][::-1][id1]
        y1 = Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'][id1],
                     fr=self.frave,
                     Lr=self.Lrave)
        x2 = self.dcpkl_Vg["Iswitch (A)"][::-1][id2]
        y2 = Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'][id2],
                     fr=self.frave,
                     Lr=self.Lrave)

        fig = plt.figure(figsize=(4, 4))
        plt.title('Ic(RF) vs Iswitch(DC)')
        plt.plot(x1 / 1e-6, y1 / 1e-6, 'o')
        plt.plot(x2 / 1e-6, y2 / 1e-6, 's')
        mylims = plt.gca().get_ylim()
        plt.plot([0, mylims[1]], [0, mylims[1]], zorder=-1)
        plt.xlim(0, mylims[1])
        plt.ylim(0, mylims[1])
        plt.locator_params(axis='both', nbins=10)
        plt.xlabel('Iswitch DC-measured (µA)')
        plt.ylabel('Ic RF-extracted (µA)')
        plt.tight_layout()
        if save:
            filename = 'plots/' + self.devpath + 'comparison_Isw-Ic' + extension + '.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()
        if not splitcnp:
            return x, y
        else:
            return (x1, x2), (y1, y2)

    def compare_Lj_Lj(self, extension='', save=True):

        id1 = self.dcpkl_Vg['Vgate (V)'] <= self.Vcnp
        id2 = self.dcpkl_Vg['Vgate (V)'] > self.Vcnp
        x1 = Ljmodel(I=0, Ic=self.dcpkl_Vg["Iswitch (A)"][::-1][id1])
        y1 = Ljmodel(I=0,
                     Ic=Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'][id1],
                                fr=self.frave,
                                Lr=self.Lrave))
        x2 = Ljmodel(I=0, Ic=self.dcpkl_Vg["Iswitch (A)"][::-1][id2])
        y2 = Ljmodel(I=0,
                     Ic=Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'][id2],
                                fr=self.frave,
                                Lr=self.Lrave))
        fig = plt.figure(figsize=(4, 4))
        plt.title('Lj(RF) vs Lj(DC)')
        plt.plot(x1 / 1e-9, y1 / 1e-9, 'o')
        plt.plot(x2 / 1e-9, y2 / 1e-9, 's')
        mylims = plt.gca().get_xlim()
        plt.plot([0, mylims[1]], [0, mylims[1]], zorder=-1)
        plt.xlim(0, mylims[1])
        plt.ylim(0, mylims[1])
        plt.locator_params(axis='both', nbins=7)
        plt.xlabel('Lj DC-extracted (nH)')
        plt.ylabel('Lj RF-measured (nH)')
        plt.tight_layout()
        if save:
            filename = 'plots/' + self.devpath + 'comparison_Lj' + extension + '.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()
        return (x1, x2), (y1, y2)

    def compare_Lj_Ic(self, extension='', save=True, logscale=True):

        id1 = self.dcpkl_Vg['Vgate (V)'] <= self.Vcnp
        id2 = self.dcpkl_Vg['Vgate (V)'] > self.Vcnp
        xDC1 = self.dcpkl_Vg["Iswitch (A)"][id1]
        yDC1 = Ljmodel(I=0, Ic=self.dcpkl_Vg["Iswitch (A)"][id1])
        xRF1 = self.dcpkl_Vg["Iswitch (A)"][::-1][id1]
        yRF1 = Ljmodel(I=0,
                       Ic=Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'][id1],
                                  fr=self.frave,
                                  Lr=self.Lrave))
        xDC2 = self.dcpkl_Vg["Iswitch (A)"][id2]
        yDC2 = Ljmodel(I=0, Ic=self.dcpkl_Vg["Iswitch (A)"][id2])
        xRF2 = self.dcpkl_Vg["Iswitch (A)"][::-1][id2]
        yRF2 = Ljmodel(I=0,
                       Ic=Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'][id2],
                                  fr=self.frave,
                                  Lr=self.Lrave))
        fig = plt.figure(figsize=(4, 4))
        plt.title('Lj(RF) vs Iswitch(DC)')
        plt.plot(xRF1 / 1e-6, yRF1 / 1e-9, '.', label='RF1')
        plt.plot(xDC1 / 1e-6, yDC1 / 1e-9, label='DC1')
        plt.plot(xRF2 / 1e-6, yRF2 / 1e-9, '.', label='RF2')
        plt.plot(xDC2 / 1e-6, yDC2 / 1e-9, label='DC2')
        if logscale:
            plt.yscale('log')
            plt.xscale('log')
        plt.xlabel('Iswitch measured (µA)')
        plt.ylabel('Lj (nH)')
        plt.legend(loc='best')
        plt.tight_layout()
        if save:
            filename = 'plots/' + self.devpath + 'comparison_Lj-Is' + extension + '.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()
        return (xDC1, xDC2), (yDC1, yDC2), (xRF1, xRF2), (yRF1, yRF2)

    def compare_Lj_I_Vg(self, extension='', save=True, logscale=True, Ib=True):

        xRF = self.dcpkl_Vg["Vgate (V)"][::-1]
        yRF = Ljmodel(I=0,
                      Ic=Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'],
                                 fr=self.frave,
                                 Lr=self.Lrave))
        xDC = self.dcpkl_Vg["Vgate (V)"]
        yDC = Ljmodel(I=0, Ic=self.dcpkl_Vg["Iswitch (A)"])
        yRF2 = Icmodel(f0=self.rfpkl_Vg['f0 (Hz)'],
                       fr=self.frave,
                       Lr=self.Lrave)
        yDC2 = self.dcpkl_Vg["Iswitch (A)"]

        fig = plt.figure()  #figsize=(4,4))
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.title('Lj vs Vgate')
        plt.plot(xRF, yRF / 1e-9, label='RF')
        plt.plot(xDC, yDC / 1e-9, label='DC')
        if logscale:
            plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('Vgate (V)')
        plt.ylabel('Lj (nH)')
        if Ib:
            mylim = plt.gca().get_ylim()
            plt.errorbar(self.fitvg,
                         Ljmodel(I=0, Ic=self.fitpkl_Ib['Ic (A)']) / 1e-9,
                         Ljmodel(I=0, Ic=self.fitpkl_Ib['Ic (A)']) / 1e-9 *
                         self.fitpkl_Ib['dIc (A)'] / self.fitpkl_Ib['Ic (A)'],
                         fmt='.',
                         label='Ib')
            plt.ylim(mylim)
        plt.legend(loc='best')

        ax2 = fig.add_subplot(gs[0, 1])
        plt.title('Ic vs Vgate')
        plt.plot(xRF, yRF2 / 1e-6, label='RF')
        plt.plot(xDC, yDC2 / 1e-6, label='DC')
        if logscale:
            plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('Vgate (V)')
        plt.ylabel('Ic (µA)')
        if Ib:
            mylim = plt.gca().get_ylim()
            plt.errorbar(self.fitvg,
                         self.fitpkl_Ib['Ic (A)'] / 1e-6,
                         self.fitpkl_Ib['dIc (A)'] / 1e-6,
                         fmt='.',
                         label='Ib')
            plt.ylim(mylim)
        plt.legend(loc='best')

        plt.tight_layout()
        if save:
            filename = 'plots/' + self.devpath + 'comparison_LjIc-Vg' + extension + '.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_to_inches='tight')
        plt.show()
        plt.close()

        if Ib:
            return xDC, yDC, yDC2, xRF, yRF, yRF2, (
                self.fitvg, Ljmodel(I=0, Ic=self.fitpkl_Ib['Ic (A)']),
                Ljmodel(I=0, Ic=self.fitpkl_Ib['Ic (A)']) *
                self.fitpkl_Ib['dIc (A)'] / self.fitpkl_Ib['Ic (A)'],
                self.fitpkl_Ib['Ic (A)'], self.fitpkl_Ib['dIc (A)'])
        else:
            return xDC, yDC, yDC2, xRF, yRF, yRF2
