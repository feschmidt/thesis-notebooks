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

```python
%run src/basemodules.py
```

## sapphire+MoRe+NbTiN leads (gJJ 2x1 shorted ref)

```python
basepath = '/home/jovyan/steelelab/measurement_data/Triton/Mark/2017-05-04_GrapheneJJ_2x1/'
datapath = glob.glob(basepath+'M5*')
[x.split('/')[-1] for x in datapath]
```

```python
myfile = glob.glob(basepath+'M56*/*.dat')
myfile
```

```python
data = stlabutils.readdata.readdat(myfile[0])
```

```python
plt.plot(data[0]['Frequency (Hz)'],data[0]['S21dB (dB)'],'-')
plt.plot(data[1]['Frequency (Hz)'],data[1]['S21dB (dB)'],'-')
```

```python
freqs, s11 = data[0]['Frequency (Hz)'],data[0]['S21re ()']+1j*data[0]['S21im ()']
s11abs = np.abs(s11)
s11ph = signal.detrend(np.unwrap(np.angle(s11)))
s11 = s11abs*np.exp(1j*s11ph)
```

```python
params,_,_,_ = stlabutils.S11fit(freqs,s11,ftype='A',doplots=True)
params
```

```python
s11back = stlabutils.S11back(freqs, params)
s11fit = stlabutils.S11func(freqs, params)
```

```python
np.exp(1j * params['theta']) 
```

```python
s11new = -s11 / s11back
s11fitnew = -s11fit / s11back
```

```python
plt.plot(freqs,abs(s11new),'.')
plt.plot(freqs,abs(s11fitnew))
```

```python
plt.plot(freqs,np.angle(s11new),'.')
plt.plot(freqs,np.angle(s11fitnew))
```

```python
df0 = (freqs-params['f0'])/params['f0']
freqs/=1e9
```

```python
fig=plt.figure(figsize=cm2inch(17,6),constrained_layout=True)
gs=fig.add_gridspec(1,3,width_ratios=(1,1,1.5))
ax1=fig.add_subplot(gs[0,0])
plt.plot(freqs,np.abs(s11new),'.')
plt.plot(freqs,np.abs(s11fitnew))

ax11=fig.add_subplot(gs[0,1])
plt.plot(freqs,np.angle(s11new)/pi,'.')
plt.plot(freqs,np.angle(s11fitnew)/pi)

axpol = fig.add_subplot(gs[0,2])
plt.plot(np.real(s11new),np.imag(s11new),'.',label='data')
plt.plot(np.real(s11fitnew),np.imag(s11fitnew),label='fit')
plt.gca().set_aspect('equal','box')
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.legend()

#for theax in [ax1,ax11]:
#    theax.legend(loc=3)
   

ax1.set_xlabel('Frequency (GHz)')
ax11.set_xlabel('Frequency (GHz)')
ax1.set_ylabel(r'$|S_{11}|$ (dB)')
ax11.set_ylabel(r'$\angle\ S_{11}$ ($\pi$)')

axpol.set_xlabel(r'$\mathcal{Re}\ S_{11}$')
axpol.set_ylabel(r'$\mathcal{Im}\ S_{11}$')
    
plt.show()
plt.close()
```

```python
params
```

## silicon+NbTiN SRON

```python
basepath = '/home/jovyan/steelelab/measurement_data/He7/Mark/180716_1807.1_NbTiN_Shunts/'
datapath = glob.glob(basepath+'*')
[x.split('/')[-1] for x in datapath]
```

```python
datapath = glob.glob(basepath+'Third_cooldown_topleft_ridge_box/*')
[x.split('/')[-1] for x in datapath]
```

```python
myfiles = glob.glob(basepath+'Third_cooldown_topleft_ridge_box/*/*.dat')
[x.split('/')[-1] for x in myfiles]
```

```python
data = stlabutils.readdata.readdat(myfiles[0])
len(data)
```

```python
plt.plot(data[0]['Frequency (Hz)'],data[0]['S21dB (dB)'],'-')
```

```python
freqs, s11 = data[0]['Frequency (Hz)'],data[0]['S21re ()']+1j*data[0]['S21im ()']
s11abs = np.abs(s11)
s11ph = signal.detrend(np.unwrap(np.angle(s11)))
s11 = s11abs*np.exp(1j*s11ph)
```

```python
params,_,_,_ = stlabutils.S11fit(freqs,s11,ftype='A',doplots=True)
params
```

```python
s11back = stlabutils.S11back(freqs, params)
s11fit = stlabutils.S11func(freqs, params)
```

```python
np.exp(1j * params['theta']) 
```

```python
s11new = -s11 / s11back
s11fitnew = -s11fit / s11back
```

```python
df0 = (freqs-params['f0'])/params['f0']
freqs/=1e9
```

```python
fig=plt.figure(figsize=cm2inch(17,6),constrained_layout=True)
gs=fig.add_gridspec(1,3,width_ratios=(1,1,1.5))
ax1=fig.add_subplot(gs[0,0])
plt.plot(freqs,np.abs(s11new),'.')
plt.plot(freqs,np.abs(s11fitnew))

ax11=fig.add_subplot(gs[0,1])
plt.plot(freqs,np.angle(s11new),'.')
plt.plot(freqs,np.angle(s11fitnew))

axpol = fig.add_subplot(gs[0,2])
plt.plot(np.real(s11new),np.imag(s11new),'.',label='data')
plt.plot(np.real(s11fitnew),np.imag(s11fitnew),label='fit')
plt.gca().set_aspect('equal','box')
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.legend()

#for theax in [ax1,ax11]:
#    theax.legend(loc=3)
   

ax1.set_xlabel('Frequency (GHz)')
ax11.set_xlabel('Frequency (GHz)')
ax1.set_ylabel(r'$|S_{11}|$')
ax11.set_ylabel(r'$\angle\ S_{11}$')

axpol.set_xlabel(r'$\mathcal{Re}\ S_{11}$')
axpol.set_ylabel(r'$\mathcal{Im}\ S_{11}$')

ax11.set_yticks([-pi,-pi/2,0,pi/2,pi])
ax11.set_yticklabels(['-$\pi$','-$\pi/2$','0','$\pi/2$','$\pi$'])

ax1.text(-0.45, 1, '(a)', transform=ax1.transAxes, fontweight='bold', va='top')
ax11.text(-0.5, 1, '(b)', transform=ax11.transAxes, fontweight='bold', va='top')
axpol.text(-0.35, 1, '(c)', transform=axpol.transAxes, fontweight='bold', va='top')

plt.savefig('plots/data_gJJ_ref_Qint.pdf',bbox_to_inches='tight',dpi=dpi)
    
plt.show()
plt.close()
```

```python
params
```
