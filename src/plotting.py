# -*- coding: utf-8 -*-
# Making overview plots for d^n(S)/df^n and other quantities

import matplotlib.pyplot as plt
import stlabutils
import numpy as np
from scipy.interpolate import interp1d


def plot3_2Dmaps(mydfs,
                 zkey1,
                 zkey2,
                 zkey3,
                 xkey='Frequency (Hz)',
                 ykey='Is (A)',
                 yscale=1e-6,
                 ylabel='Bias current (µA)',
                 applysteps=['rotate_ccw'],
                 cmap='RdBu',
                 supertitle=None,
                 ylim3_2D=None,
                 colorbar=True,
                 scriptname=None):

    mymtx1 = stlabutils.framearr_to_mtx(mydfs, zkey1, xkey=xkey, ykey=ykey)
    mymtx2 = stlabutils.framearr_to_mtx(mydfs, zkey2, xkey=xkey, ykey=ykey)
    mymtx3 = stlabutils.framearr_to_mtx(mydfs, zkey3, xkey=xkey, ykey=ykey)

    if applysteps:
        for mymtx in [mymtx1, mymtx2, mymtx3]:
            for thestep in applysteps:
                mymtx.applystep(thestep)
    else:
        print('No steps to apply. Continuing...')

    wbval = (0.1, 0.1)
    lims1 = np.percentile(mymtx1.pmtx.values, (wbval[0], 100 - wbval[1]))
    vmin1 = lims1[0]
    vmax1 = lims1[1]
    extents1 = mymtx1.getextents()

    lims2 = np.percentile(mymtx2.pmtx.values, (wbval[0], 100 - wbval[1]))
    vmin2 = lims2[0]
    vmax2 = lims2[1]
    extents2 = mymtx2.getextents()

    lims3 = np.percentile(mymtx3.pmtx.values, (wbval[0], 100 - wbval[1]))
    vmin3 = lims3[0]
    vmax3 = lims3[1]
    extents3 = mymtx3.getextents()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    plt.sca(ax1)
    x = mymtx1.pmtx.axes[1] / yscale
    y = mymtx1.pmtx.axes[0] / 1e9
    X, Y = np.meshgrid(x, y)
    Z = mymtx1.pmtx.values
    plt.pcolormesh(X,
                   Y,
                   Z,
                   cmap=cmap,
                   vmin=vmin1,
                   vmax=vmax1,
                   linewidth=0,
                   rasterized=True)
    # plt.imshow(mymtx1.pmtx, aspect='auto', cmap=cmap, extent=(
    #     extents1[0]/yscale, extents1[1]/yscale, extents1[2]/1e9, extents1[3]/1e9), vmin=vmin1, vmax=vmax1)
    # plt.xlim(0,8)
    # cbar.set_label('S11dB (dB)')

    plt.sca(ax2)
    x = mymtx2.pmtx.axes[1] / yscale
    y = mymtx2.pmtx.axes[0] / 1e9
    X, Y = np.meshgrid(x, y)
    Z = mymtx2.pmtx.values
    plt.pcolormesh(X,
                   Y,
                   Z,
                   cmap=cmap,
                   vmin=vmin2,
                   vmax=vmax2,
                   linewidth=0,
                   rasterized=True)
    # plt.imshow(mymtx2.pmtx, aspect='auto', cmap=cmap, extent=(
    #     extents2[0]/yscale, extents2[1]/yscale, extents2[2]/1e9, extents2[3]/1e9), vmin=vmin2, vmax=vmax2)
    # plt.xlim(0,8)
    # cbar.set_label('S11dB (dB)')

    plt.sca(ax3)
    x = mymtx3.pmtx.axes[1] / yscale
    y = mymtx3.pmtx.axes[0] / 1e9
    X, Y = np.meshgrid(x, y)
    Z = mymtx3.pmtx.values
    plt.pcolormesh(X,
                   Y,
                   Z,
                   cmap=cmap,
                   vmin=vmin3,
                   vmax=vmax3,
                   linewidth=0,
                   rasterized=True)
    # plt.imshow(mymtx3.pmtx, aspect='auto', cmap=cmap, extent=(
    #     extents3[0]/yscale, extents3[1]/yscale, extents3[2]/1e9, extents3[3]/1e9), vmin=vmin3, vmax=vmax3)
    # plt.xlim(0,8)
    # cbar.set_label('S11dB (dB)')

    for ax, zkey in zip([ax1, ax2, ax3], [zkey1, zkey2, zkey3]):
        ax.set_xlabel(ylabel)
        ax.set_ylabel('Frequency (GHz)')
        ax.set_title(zkey)
        plt.sca(ax)
        if colorbar:
            _ = plt.colorbar()

    if ylim3_2D:
        [ax.set_ylim(ylim3_2D[0], ylim3_2D[1]) for ax in [ax1, ax2, ax3]]
    if scriptname:
        fig.text(.5, .05, scriptname, ha='center')
    if supertitle:
        plt.suptitle(supertitle)
    else:
        print('No supertitle provided. Continuing...')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close()


def plot_four_overview(data,
                       key='dSdf',
                       unit='(1/Hz)',
                       savefig=False,
                       showfig=True,
                       ii=0,
                       scriptname=None,
                       ykey='Is (A)',
                       yscale=1e-6,
                       yval='I0',
                       yunit='uA'):

    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 8))
    plt.sca(ax1[0])
    plt.plot(data['Frequency (Hz)'],
             data[key + 'abs ' + unit],
             label='data abs')
    plt.plot(data['Frequency (Hz)'],
             data[key + 'fitabs ' + unit],
             label='fit abs')
    plt.legend()

    plt.sca(ax1[1])
    plt.plot(data['Frequency (Hz)'],
             data[key + 'fitnobgabs ' + unit],
             label='nobg abs')
    plt.legend()

    plt.sca(ax2[0])
    plt.plot(data['Frequency (Hz)'], data[key + 'fitre ' + unit], label='re')
    plt.plot(data['Frequency (Hz)'], data[key + 'fitim ' + unit], label='im')
    plt.legend()

    plt.sca(ax2[1])
    plt.plot(data['Frequency (Hz)'],
             data[key + 'fitnobgre ' + unit],
             label='nobg re')
    plt.plot(data['Frequency (Hz)'],
             data[key + 'fitnobgim ' + unit],
             label='nobg im')

    plt.legend()
    plt.suptitle(key + ' ' + unit +
                 f', {yval}={abs(data[ykey][0] / yscale)}{yunit}')
    if scriptname:
        fig.text(.5, .05, scriptname, ha='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if savefig:
        plt.savefig('plots/sensitivity_estimation/' + key + '_plots/' + key +
                    f'_plots_{ii + 1}.png',
                    bbox_to_inches='tight')
    if showfig:
        plt.show()
    plt.close()


def plotall_four_overview(data, key='dSdf', unit='(1/Hz)'):

    doit = input('Do you want to plot and save all of these figures? y/n\n')

    if doit == 'y':
        print('Doing it...')
        for ii, myblock in enumerate(data):
            plot_four_overview(myblock,
                               key,
                               unit,
                               savefig=True,
                               showfig=False,
                               ii=ii)

    else:
        print('Continuing without plotting and saving...')
