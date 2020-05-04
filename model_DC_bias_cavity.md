---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Model DC bias cavity


All theory plots for the DC bias cavity circuit model

```python
%load_ext autoreload
%autoreload 2
```

```python
%run src/basemodules.py
```

## f0 vs Lj

```python
fr = 8e9
Lr = 3e-9
Cs = 30e-12
```

```python
Lj = np.logspace(-12, -6.5, 401)
```

```python
w = 2 * pi * np.linspace(3e9, 8.4e9, 10001)
wr = 2 * pi * fr
```

```python
f0_appr = fr / 2 * (1 + 1 / (1 + Lj / (Lr / 2)))
```

```python
from src.model import gJJ_analytical, gJJ_analytical_simple
from scipy.optimize import brentq, root
```

```python
solfull = np.array(
    [brentq(gJJ_analytical, 3e9 * 2 * pi, wr, args=(x, 1, wr))
     for x in Lj]) / 2 / pi
```

```python
plt.plot(Lj, f0_appr, label='approximation')
plt.plot(Lj, solfull, label='exact')
plt.xscale('log')
plt.legend()
```

```python
rfmodel = plt.imread('plots/rfmodel.png')
```

```python
fig = plt.figure(figsize=cm2inch(17, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
ax1 = fig.add_subplot(gs[0, 0])
plt.imshow(rfmodel)
plt.axis('off')
ax2 = fig.add_subplot(gs[0, 1])
plt.plot(Lj, f0_appr / 1e9, label='approximation')
plt.xscale('log')
plt.xlabel(r'$L_J$ (H)')
plt.ylabel(r'$f_0$ (GHz)')
plt.savefig('plots/model_DC_bias_cavity_params.pdf', bbox_to_inches='tight')
plt.show()
plt.close()
```

```python
np.exp(2 * 0.006073 * 6119e-6)
```

```python
np.log10(Phi0 / 2 / pi / 1e-9)
```

## S11

```python
def S11(ki, ke, w0, w):
    Delta = w - w0
    return -1 + 2 * ke / (ki + ke + 2j * Delta)
```

```python
ki = 2 * pi * 200e3
ke = 2 * pi * 1e6
f0 = 8e9
w0 = 2 * pi * f0
```

```python
print(f'Internal Q: {w0/ki:.0f}')
print(f'External Q: {w0/ki:.0f}, {w0/(0.1*ki):.0f}, {w0/(10*ki):.0f}')
```

```python
df = 10e6
w = 2 * pi * np.linspace(f0 - df, f0 + df, 801)
```

```python
fig = plt.figure(figsize=cm2inch(17, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.5))
ax1 = fig.add_subplot(gs[0, 0])
plt.plot((w) / w0,
         #20 * np.log10(np.abs(S11(ki, ki, w0, w))),
         np.abs(S11(ki, ki, w0, w)),
         label='$\kappa_e=\kappa_i$')
plt.plot((w) / w0,
         #20 * np.log10(np.abs(S11(ki, ki / 10, w0, w))),
         np.abs(S11(ki, ki / 10, w0, w)),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         #20 * np.log10(np.abs(S11(ki, ki * 10, w0, w))),
         np.abs(S11(ki, ki * 10, w0, w)),
         label='$\kappa_e=10\kappa_i$')
#plt.ylim(-5,.4)

ax11 = fig.add_subplot(gs[0, 1])
plt.plot((w) / w0,
         np.angle(S11(ki, ki, w0, w)) / pi,
         label='$\kappa_e=\kappa_i$')
plt.plot((w) / w0,
         np.angle(S11(ki, ki / 10, w0, w)) / pi,
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         np.angle(S11(ki, ki * 10, w0, w)) / pi,
         label='$\kappa_e=10\kappa_i$')

axpol = fig.add_subplot(gs[0, 2])
plt.plot(np.real(S11(ki, ki, w0, w)),
         np.imag(S11(ki, ki, w0, w)),
         label='$\kappa_e=\kappa_i$')
plt.plot(np.real(S11(ki, ki / 10, w0, w)),
         np.imag(S11(ki, ki / 10, w0, w)),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot(np.real(S11(ki, ki * 10, w0, w)),
         np.imag(S11(ki, ki * 10, w0, w)),
         label='$\kappa_e=10\kappa_i$')
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.legend(loc=1)
axpol.set_xticks([-1,-0.5,0,0.5,1])
axpol.set_yticks([-1,-0.5,0,0.5,1])

#for theax in [ax1,ax11]:
#    theax.legend(loc=3)

ax1.set_xlabel('$\delta\omega/\omega_0$')
ax11.set_xlabel('$\delta\omega/\omega_0$')
ax1.set_ylabel(r'$|S_{11}|$')
ax11.set_ylabel(r'$\angle\ S_{11}$ ($\pi$)')

axpol.set_xlabel(r'$\mathcal{Re}\ S_{11}$')
axpol.set_ylabel(r'$\mathcal{Im}\ S_{11}$')

### inset
axins = ax1.inset_axes([0.55, 0.05, 0.4, 0.3])
axins.plot((w) / w0,
           #20 * np.log10(np.abs(S11(ki, ki, w0, w))),
           np.abs(S11(ki, ki, w0, w)),
           label='$\kappa_e=\kappa_i$')
axins.plot((w) / w0,
           #20 * np.log10(np.abs(S11(ki, ki / 10, w0, w))),
           np.abs(S11(ki, ki / 10, w0, w)),
           'C1',
           label='$\kappa_e=0.1\kappa_i$')
axins.plot((w) / w0,
           #20 * np.log10(np.abs(S11(ki, ki * 10, w0, w))),
           np.abs(S11(ki, ki * 10, w0, w)),
           'C2',
           label='$\kappa_e=10\kappa_i$')
# sub region of the original image
x1, x2, y1, y2 = 0.9999, 1.0001, 0.75, 1.05
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([])
axins.set_yticks([])

ax1.indicate_inset_zoom(axins)

ax1.text(-0.5, 1, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')
ax11.text(-0.55, 1, '(b)', transform=ax11.transAxes, fontweight='bold', va='top')
axpol.text(-0.35, 1, '(c)', transform=axpol.transAxes, fontweight='bold', va='top')

plt.savefig('plots/model_DC_bias_cavity_coupling.pdf', bbox_to_inches='tight')
plt.show()
plt.close()
```

```python hide_input=true
fig = plt.figure(figsize=cm2inch(17, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.5))
ax1 = fig.add_subplot(gs[0, 0])
plt.plot((w) / w0,
         20 * np.log10(np.abs(S11(ki, ki, w0, w))),
         label='$\kappa_e=\kappa_i$')
plt.plot((w) / w0,
         20 * np.log10(np.abs(S11(ki, ki / 10, w0, w))),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         20 * np.log10(np.abs(S11(ki, ki * 10, w0, w))),
         label='$\kappa_e=10\kappa_i$')

ax11 = fig.add_subplot(gs[0, 1])
plt.plot((w) / w0,
         np.angle(S11(ki, ki, w0, w)) / pi,
         label='$\kappa_e=\kappa_i$')
plt.plot((w) / w0,
         np.angle(S11(ki, ki / 10, w0, w)) / pi,
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         np.angle(S11(ki, ki * 10, w0, w)) / pi,
         label='$\kappa_e=10\kappa_i$')

axpol = fig.add_subplot(gs[0, 2], projection='polar')
plt.plot(np.angle(S11(ki, ki, w0, w)),
         np.abs(S11(ki, ki, w0, w)),
         label='$\kappa_e=\kappa_i$')
plt.plot(np.angle(S11(ki, ki / 10, w0, w)),
         np.abs(S11(ki, ki / 10, w0, w)),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot(np.angle(S11(ki, ki * 10, w0, w)),
         np.abs(S11(ki, ki * 10, w0, w)),
         label='$\kappa_e=10\kappa_i$')
#plt.gca().set_aspect('equal', 'box')
#plt.xlim(-1.1, 1.1)
#plt.ylim(-1.1, 1.1)
plt.legend(loc=4)

#for theax in [ax1,ax11]:
#    theax.legend(loc=3)

ax1.set_xlabel('$\delta\omega/\omega_0$')
ax11.set_xlabel('$\delta\omega/\omega_0$')
ax1.set_ylabel(r'$|S_{11}|$ (dB)')
ax11.set_ylabel(r'$\angle\ S_{11}$ ($\pi$)')

#axpol.set_xlabel(r'$\mathcal{Re}\ S_{11}$')
#axpol.set_ylabel(r'$\mathcal{Im}\ S_{11}$')

plt.show()
plt.close()
```

```python hide_input=true
fig = plt.figure(figsize=cm2inch(17, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=(1, 1.5))
ax1 = fig.add_subplot(gs[0, 0])
plt.plot((w) / w0,
         20 * np.log10(np.abs(S11(ki, ki, w0, w))),
         label='$\kappa_e=\kappa_i$')
plt.plot((w) / w0,
         20 * np.log10(np.abs(S11(ki, ki / 10, w0, w))),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         20 * np.log10(np.abs(S11(ki, ki * 10, w0, w))),
         label='$\kappa_e=10\kappa_i$')

ax11 = fig.add_subplot(gs[1, 0])
plt.plot((w) / w0,
         np.angle(S11(ki, ki, w0, w)) / pi,
         label='$\kappa_e=\kappa_i$')
plt.plot((w) / w0,
         np.angle(S11(ki, ki / 10, w0, w)) / pi,
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         np.angle(S11(ki, ki * 10, w0, w)) / pi,
         label='$\kappa_e=10\kappa_i$')

axpol = fig.add_subplot(gs[:, 1])
plt.plot(np.real(S11(ki, ki, w0, w)),
         np.imag(S11(ki, ki, w0, w)),
         label='$\kappa_e=\kappa_i$')
plt.plot(np.real(S11(ki, ki / 10, w0, w)),
         np.imag(S11(ki, ki / 10, w0, w)),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot(np.real(S11(ki, ki * 10, w0, w)),
         np.imag(S11(ki, ki * 10, w0, w)),
         label='$\kappa_e=10\kappa_i$')
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.legend(loc=1)

for theax in [ax1, ax11]:
    theax.legend(loc=3)

for theax in [ax1]:
    theax.set_xticklabels([])

ax11.set_xlabel('$\delta\omega/\omega_0$')
ax1.set_ylabel(r'$|S_{11}|$ (dB)')
ax11.set_ylabel(r'$\angle\ S_{11}$ ($\pi$)')

axpol.set_xlabel(r'$\mathcal{Re}\ S_{11}$')
axpol.set_ylabel(r'$\mathcal{Im}\ S_{11}$')

plt.show()
plt.close()
```

```python hide_input=true
fig = plt.subplot(projection='polar')
plt.plot(np.angle(S11(ki, ki, w0, w)),
         np.abs(S11(ki, ki, w0, w)),
         label='$\kappa_e=\kappa_i$')
plt.plot(np.angle(S11(ki, ki / 10, w0, w)),
         np.abs(S11(ki, ki / 10, w0, w)),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot(np.angle(S11(ki, ki * 10, w0, w)),
         np.abs(S11(ki, ki * 10, w0, w)),
         label='$\kappa_e=10\kappa_i$')
plt.legend(loc=4)
plt.show()
plt.close()
```

```python
plt.plot((w) / w0,
         20 * np.log10(np.abs(S11(ki, ki / 10, w0, w))),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         20 * np.log10(np.abs(S11(ki, ki * 10, w0, w))),
         label='$\kappa_e=10\kappa_i$')
```

```python
min(np.abs(S11(ki, ki / 5, w0, w))), min(np.abs(S11(ki, ki * 5, w0, w)))
```

```python
plt.plot((w) / w0,
         np.real(S11(ki, ki / 10, w0, w)),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         np.real(S11(ki, ki * 10, w0, w)),
         label='$\kappa_e=10\kappa_i$')
plt.plot((w) / w0,
         np.imag(S11(ki, ki / 10, w0, w)),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         np.imag(S11(ki, ki * 10, w0, w)),
         label='$\kappa_e=10\kappa_i$')
plt.legend()
```

## S21

```python
def S21(ki, ke, w0, w):
    Delta = w - w0
    return -ke / (ki + ke + 2j * Delta)
```

```python
ki = 2 * pi * 200e3
ke = 2 * pi * 1e6
f0 = 8e9
w0 = 2 * pi * f0
```

```python
print(f'Internal Q: {w0/ki:.0f}')
print(f'External Q: {w0/ki:.0f}, {w0/(0.1*ki):.0f}, {w0/(10*ki):.0f}')
```

```python
df = 10e6
w = 2 * pi * np.linspace(f0 - df, f0 + df, 1001)
```

```python
fig = plt.figure(figsize=cm2inch(17, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.5))
ax1 = fig.add_subplot(gs[0, 0])
plt.plot((w) / w0,
         20 * np.log10(np.abs(S21(ki, ki, w0, w))),
         label='$\kappa_e=\kappa_i$')
plt.plot((w) / w0,
         20 * np.log10(np.abs(S21(ki, ki / 10, w0, w))),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         20 * np.log10(np.abs(S21(ki, ki * 10, w0, w))),
         label='$\kappa_e=10\kappa_i$')

ax11 = fig.add_subplot(gs[0, 1])
plt.plot((w) / w0,
         np.angle(S21(ki, ki, w0, w)) / pi,
         label='$\kappa_e=\kappa_i$')
plt.plot((w) / w0,
         np.angle(S21(ki, ki / 10, w0, w)) / pi,
         label='$\kappa_e=0.1\kappa_i$')
plt.plot((w) / w0,
         np.angle(S21(ki, ki * 10, w0, w)) / pi,
         label='$\kappa_e=10\kappa_i$')

axpol = fig.add_subplot(gs[0, 2])
plt.plot(np.real(S21(ki, ki, w0, w)),
         np.imag(S21(ki, ki, w0, w)),
         label='$\kappa_e=\kappa_i$')
plt.plot(np.real(S21(ki, ki / 10, w0, w)),
         np.imag(S21(ki, ki / 10, w0, w)),
         label='$\kappa_e=0.1\kappa_i$')
plt.plot(np.real(S21(ki, ki * 10, w0, w)),
         np.imag(S21(ki, ki * 10, w0, w)),
         label='$\kappa_e=10\kappa_i$')
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.legend(loc=1)

#for theax in [ax1,ax11]:
#    theax.legend(loc=3)

ax1.set_xlabel('$\delta\omega/\omega_0$')
ax11.set_xlabel('$\delta\omega/\omega_0$')
ax1.set_ylabel(r'$|S_{21}|$ (dB)')
ax11.set_ylabel(r'$\angle\ S_{21}$ ($\pi$)')

axpol.set_xlabel(r'$\mathcal{Re}\ S_{21}$')
axpol.set_ylabel(r'$\mathcal{Im}\ S_{21}$')

plt.show()
plt.close()
```

## QUCS sims

```python
myfiles = glob.glob('qucs/*.dat')
myfiles
```

### TL short

```python
data = stlabutils.readdata.readQUCS('qucs/thesis-TL_short.dat')
```

```python
freqs = data[0]['indep_frequency']
s11 = data[0]['dep_S[1,1]']
```

```python
params, _, _, _ = stlabutils.S11fit(freqs, s11, ftype='A', doplots=True)
s11fit = stlabutils.S11func(freqs, params)
params
```

```python
plt.plot(freqs, np.abs(s11), 'o')
plt.plot(freqs, np.abs(s11fit))
params['f0'] / params['Qint']
```

```python
pickle.dump({
    'params': params
}, open('qucs/thesis-TL_short.pkl', 'wb'))
```

### TL short, sweep alpha

```python
data = stlabutils.readdata.readQUCS('qucs/thesis-TL_short_alphasweep.dat')
```

```python
freqs = data[0]['indep_frequency']
alpha = data[0]['indep_alpha']
s11 = data[0]['dep_S[1,1]']
```

```python
s11 = s11.reshape(len(alpha), len(freqs))
Xfreqs, Yalphas = np.meshgrid(freqs, alpha)
```

```python
plt.pcolormesh(Xfreqs / 1e9, Yalphas, abs(s11))
```

```python
fits, pars = [], []
for i, z in enumerate(s11):
    print(f'############### {i+1}/{len(s11)}')
    params, _, _, _ = stlabutils.S11fit(freqs, z, ftype='A', doplots=False)
    s11fit = stlabutils.S11func(freqs, params)
    fits.append(s11fit)
    pars.append(params)
```

```python
plt.plot(alpha[1:], [x['Qint'] for x in pars[1:]], 'o',label='Qi')
plt.plot(alpha[1:], [x['Qext'] for x in pars[1:]], 'o',label='Qe')
plt.yscale('log')
plt.legend()
tt = 5
plt.axvline(alpha[tt], c='k')
pars[tt]['Qext'], pars[tt]['Qint'], alpha[tt]
```

```python
plt.plot(alpha, [x['f0'] for x in pars], 'o')
```

```python
pickle.dump({
    'alpha': alpha,
    'pars': pars
}, open('qucs/thesis-TL_short_alphasweep.pkl', 'wb'))
```

### TL RSCJ, sweep Rsg

```python
data = stlabutils.readdata.readQUCS('qucs/thesis-TL_RCSJ_Rsgsweep.dat')
```

```python
freqs = data[0]['indep_frequency']
Rsg = data[0]['indep_Rsubgap']
s11 = data[0]['dep_S[1,1]']
```

```python
s11 = s11.reshape(len(Rsg), len(freqs))
Xfreqs, Yrsg = np.meshgrid(freqs, Rsg)
```

```python
plt.pcolormesh(Xfreqs / 1e9, Yrsg, abs(s11), vmin=0.9, vmax=1)
plt.yscale('log')
plt.colorbar()
```

```python
plt.pcolormesh(Xfreqs / 1e9,
               Yrsg,
               signal.detrend(np.unwrap(np.angle(s11))),
               cmap='vlag',
               vmin=-0.01,
               vmax=0.01)
plt.yscale('log')
plt.colorbar()
```

```python
for rr, zz in zip(Rsg[::20], s11[::20]):
    plt.plot(freqs, abs(zz), label=f"{rr:.0E}")

plt.ylim(0.9, 1)
plt.legend()
```

```python
for rr, zz in zip(Rsg[::20], s11[::20]):
    plt.plot(freqs, signal.detrend(np.unwrap(np.angle(zz))), label=f"{rr:.0E}")

#plt.ylim(0.9,1)
plt.legend()
```

```python
fits, pars = [], []
for i, z in enumerate(s11):
    print(f'\n############### {i+1}/{len(s11)}\n')
    if i == 0:
        params, _, _, _ = stlabutils.S11fit(freqs, z, ftype='A', doplots=False)
    else:
        params, _, _, _ = stlabutils.S11fit(freqs,
                                            z,
                                            ftype='A',
                                            doplots=False,
                                            reusefitpars=True,
                                            oldpars=params)
    s11fit = stlabutils.S11func(freqs, params)
    fits.append(s11fit)
    pars.append(params)
```

```python
plt.plot(Rsg, [x['Qint'] for x in pars])
plt.plot(Rsg, [x['Qext'] for x in pars])
plt.yscale('log')
plt.xscale('log')
```

```python
plt.plot(Rsg, [x['f0'] / x['Qint'] for x in pars])
plt.plot(Rsg, [x['f0'] / x['Qext'] for x in pars])
plt.xscale('log')
plt.yscale('log')
```

```python
plt.plot(Rsg, [x['f0'] for x in pars])
plt.xscale('log')
```

```python
pickle.dump({
    'Rsg': Rsg,
    'pars': pars
}, open('qucs/thesis-TL_RCSJ_Rsgsweep.pkl', 'wb'))
```

### TL RSCJ, sweep LJ

```python
data = stlabutils.readdata.readQUCS('qucs/thesis-TL_RCSJ_LJsweep.dat')
```

```python
freqs = data[0]['indep_frequency']
#LJ = Phi0 / 2 / pi / data[0]['indep_Ic']
LJ = data[0]['indep_JJL']
s11 = data[0]['dep_S[1,1]']
```

```python
s11 = s11.reshape(len(LJ), len(freqs))
Xfreqs, Ylj = np.meshgrid(freqs, LJ)
```

```python
plt.pcolormesh(Xfreqs / 1e9, Ylj, np.abs(s11))  #,vmin=0.9,vmax=1)
plt.yscale('log')
plt.colorbar()
```

```python
plt.pcolormesh(Xfreqs / 1e9, Ylj,
               np.gradient(np.angle(s11))[0],cmap='vlag')  #,vmin=0.9,vmax=1)
plt.yscale('log')
plt.colorbar()
plt.show()
plt.close()
```

```python
plt.plot(freqs / 1e9, abs(s11[0]), 'o')
plt.xlim(7.315,7.323)
```

```python
f0s_Lj = []
for i, z in enumerate(s11):
    idx = np.argmin(abs(z))
    f0s_Lj.append(freqs[idx])
f0s_Lj = np.array(f0s_Lj)
```

```python
plt.plot(LJ, f0s_Lj)
plt.xscale('log')
```

```python
fits, parsLj = [], []
for i, z in enumerate(s11):
    print(f'\n############### {i+1}/{len(s11)}\n')
    if i == 0:
        params, _, _, _ = stlabutils.S11fit(freqs, z, ftype='A', doplots=False)
    else:
        params, _, _, _ = stlabutils.S11fit(freqs,
                                            z,
                                            ftype='A',
                                            doplots=False,
                                            reusefitpars=True,
                                            oldpars=params)
    s11fit = stlabutils.S11func(freqs, params)
    fits.append(s11fit)
    parsLj.append(params)
```

```python
plt.plot(LJ, [x['Qint'] for x in parsLj])
plt.plot(LJ, [x['Qext'] for x in parsLj])
plt.yscale('log')
plt.xscale('log')
```

```python
plt.plot(LJ, [x['f0'] / x['Qint'] for x in parsLj])
plt.plot(LJ, [x['f0'] / x['Qext'] for x in parsLj])
plt.xscale('log')
plt.yscale('log')
```

```python
plt.plot(LJ, [x['f0'] for x in parsLj])
plt.xscale('log')
```

```python
pickle.dump({
    'LJ': LJ,
    'pars': parsLj,
    'f0s_Lj': f0s_Lj
}, open('qucs/thesis-TL_RCSJ_LJsweep.pkl', 'wb'))
```

# final plot


## RFmodel + Ljsweep + Rsgsweep

```python
LJsweep = pickle.load(open('qucs/thesis-TL_RCSJ_LJsweep.pkl', 'rb'))
LJ, parsLj, f0s_Lj = LJsweep['LJ'], LJsweep['pars'], LJsweep['f0s_Lj']

Rsgsweep = pickle.load(open('qucs/thesis-TL_RCSJ_Rsgsweep.pkl', 'rb'))
Rsg, parsRsg = Rsgsweep['Rsg'], Rsgsweep['pars']
```

```python
paramsshort = pickle.load(open('qucs/thesis-TL_short.pkl', 'rb'))
fr = paramsshort['params']['f0']
```

```python
Lj = np.logspace(-12, -6.5, 401)
Lr = 3e-9
f0_appr = fr / 2 * (1 + 1 / (1 + Lj / (Lr / 2)))
```

```python
rfmodel = plt.imread('plots/rfmodel.png')
```

```python
fig = plt.figure(figsize=cm2inch(17, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2)  #,width_ratios=[1.5,1])

ax1 = fig.add_subplot(gs[0, 0])
plt.imshow(rfmodel)
plt.axis('off')

ax2 = fig.add_subplot(gs[1, 0])
#plt.plot(LJ, np.array([x['f0'] for x in parsLj])/1e9)
plt.plot(LJ, f0s_Lj / 1e9)
plt.xscale('log')
plt.xlabel(r'$L_J$ (H)')
plt.ylabel(r'$f_0$ (GHz)')

ax3 = fig.add_subplot(gs[1, 1])
plt.plot(Rsg, [x['f0'] / 1e9 for x in parsRsg])
plt.xscale('log')
plt.xlabel(r'$R_{\rm sg}$ ($\Omega$)')
plt.ylabel(r'$f_0$ (GHz)')

ax4 = fig.add_subplot(gs[0, 1])
plt.plot(Rsg, [x['Qint'] / 1e3 for x in parsRsg])
plt.xscale('log')
plt.gca().set_xticklabels([])

plt.ylabel(r'$Q_i$ ($10^3$)')

ax1.text(-0.25, 1, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')
ax2.text(-0.16, 1, '(b)', transform=ax2.transAxes, fontweight='bold', va='top')
ax3.text(-0.28, 1, '(c)', transform=ax3.transAxes, fontweight='bold', va='top')
ax4.text(-0.28, 1, '(d)', transform=ax4.transAxes, fontweight='bold', va='top')

plt.savefig('plots/model_DC_bias_cavity_params_RCSJ.pdf',
            bbox_to_inches='tight',
            dpi=dpi)
#plt.savefig('plots/model_DC_bias_cavity_params_RCSJ.png',bbox_to_inches='tight')
plt.show()
plt.close()
```

```python

```
