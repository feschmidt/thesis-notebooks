"""
Modules needed for data analysis.
We import them in the top-level cell of all notebooks via `%run src/basemodules.py`.
These modules then become globally available in the notebook.
"""

# Python modules
import copy
import glob
import lmfit
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy import signal
from scipy.interpolate import interp1d
import scipy.constants
from scipy.constants import hbar, pi, Planck, e, k
import stlabutils

echarge = e
Phi0 = Planck / (2 * echarge)
Rk = Planck / (echarge**2)
kB = k

# Self-written modules
from src.model import f0 as f0model
from src.model import Lj as Ljmodel
from src.model import lossrate, Icmodel, Anh_seriesLCJ, Anh_Zhou, Anh_Pogo, CPR, Lj_of_I, f0_of_I
from src.plotting import plot_four_overview, plotall_four_overview, plot3_2Dmaps
from src.myfuncs import func_PdBmtoV
from src.comparison import Comparison
from src.residuals import ResidualOverview, ResidualWrapper, EmceeWrapper, DataWrapper, DataBestTau

# Plot parameters
plotall = True
overview_plot = True

cmap = matplotlib.cm.get_cmap('binary')

import seaborn as sns
sns.set()
sns.set_style('white')
sns.set_style('ticks')
plt.style.use('src/my_rcparams.mplstyle')

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


dpi = 1000

# IPython modules
from IPython.display import Image, display

# +
# Path to data
#basepath = '/home/jovyan/steelelab/measurement_data/Triton/Mark/'

# +
# Print file contents
#with open('src/basemodules.py') as f:
#    print(f.read())
