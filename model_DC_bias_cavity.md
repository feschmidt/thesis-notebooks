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
Lj = np.logspace(-12,-7,401)
```

```python
w = 2 * pi * np.linspace(3e9, 8.4e9, 10001)
wr = 2*pi*fr
```

```python
f0_appr = fr / 2 * (1 + 1 / (1 + Lj / (Lr / 2)))
```

```python
from src.model import gJJ_analytical, gJJ_analytical_simple
from scipy.optimize import brentq, root
```

```python
solfull = np.array([brentq(gJJ_analytical,3e9*2*pi,wr,args=(x,1,wr)) for x in Lj])/2/pi
```

```python
plt.plot(Lj,f0_appr,label='approximation')
plt.plot(Lj,solfull,label='exact')
plt.xscale('log')
plt.legend()
```

## S11

```python
def S11(ki,ke,w0,w):
    Delta = w-w0
    return -1+2*ke/(ki+ke+2j*Delta)
```

```python
ki=2*pi*200e3
ke=2*pi*1e6
f0 = 8e9
w0=2*pi*f0
```

```python
print(f'Internal Q: {w0/ki:.0f}')
print(f'External Q: {w0/ki:.0f}, {w0/(0.1*ki):.0f}, {w0/(10*ki):.0f}')
```

```python
df=10e6
w=2*pi*np.linspace(f0-df,f0+df,1001)
```

```python
fig=plt.figure(figsize=cm2inch(17,8),constrained_layout=True)
gs=fig.add_gridspec(2,2,width_ratios=(1,1.5))
ax1=fig.add_subplot(gs[0,0])
plt.plot((w)/w0,20*np.log10(np.abs(S11(ki,ki,w0,w))),label='$\kappa_e=\kappa_i$')
plt.plot((w)/w0,20*np.log10(np.abs(S11(ki,ki/10,w0,w))),label='$\kappa_e=0.1\kappa_i$')
plt.plot((w)/w0,20*np.log10(np.abs(S11(ki,ki*10,w0,w))),label='$\kappa_e=10\kappa_i$')

ax11=fig.add_subplot(gs[1,0])
plt.plot((w)/w0,np.angle(S11(ki,ki,w0,w))/pi,label='$\kappa_e=\kappa_i$')
plt.plot((w)/w0,np.angle(S11(ki,ki/10,w0,w))/pi,label='$\kappa_e=0.1\kappa_i$')
plt.plot((w)/w0,np.angle(S11(ki,ki*10,w0,w))/pi,label='$\kappa_e=10\kappa_i$')

axpol = fig.add_subplot(gs[:,1])
plt.plot(np.real(S11(ki,ki,w0,w)),np.imag(S11(ki,ki,w0,w)),label='$\kappa_e=\kappa_i$')
plt.plot(np.real(S11(ki,ki/10,w0,w)),np.imag(S11(ki,ki/10,w0,w)),label='$\kappa_e=0.1\kappa_i$')
plt.plot(np.real(S11(ki,ki*10,w0,w)),np.imag(S11(ki,ki*10,w0,w)),label='$\kappa_e=10\kappa_i$')
plt.gca().set_aspect('equal','box')
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.legend(loc=1)

for theax in [ax1,ax11]:
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

```python
fig=plt.figure(figsize=cm2inch(17,6),constrained_layout=True)
gs=fig.add_gridspec(1,3,width_ratios=(1,1,1.5))
ax1=fig.add_subplot(gs[0,0])
plt.plot((w)/w0,20*np.log10(np.abs(S11(ki,ki,w0,w))),label='$\kappa_e=\kappa_i$')
plt.plot((w)/w0,20*np.log10(np.abs(S11(ki,ki/10,w0,w))),label='$\kappa_e=0.1\kappa_i$')
plt.plot((w)/w0,20*np.log10(np.abs(S11(ki,ki*10,w0,w))),label='$\kappa_e=10\kappa_i$')

ax11=fig.add_subplot(gs[0,1])
plt.plot((w)/w0,np.angle(S11(ki,ki,w0,w))/pi,label='$\kappa_e=\kappa_i$')
plt.plot((w)/w0,np.angle(S11(ki,ki/10,w0,w))/pi,label='$\kappa_e=0.1\kappa_i$')
plt.plot((w)/w0,np.angle(S11(ki,ki*10,w0,w))/pi,label='$\kappa_e=10\kappa_i$')

axpol = fig.add_subplot(gs[0,2])
plt.plot(np.real(S11(ki,ki,w0,w)),np.imag(S11(ki,ki,w0,w)),label='$\kappa_e=\kappa_i$')
plt.plot(np.real(S11(ki,ki/10,w0,w)),np.imag(S11(ki,ki/10,w0,w)),label='$\kappa_e=0.1\kappa_i$')
plt.plot(np.real(S11(ki,ki*10,w0,w)),np.imag(S11(ki,ki*10,w0,w)),label='$\kappa_e=10\kappa_i$')
plt.gca().set_aspect('equal','box')
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.legend(loc=1)

#for theax in [ax1,ax11]:
#    theax.legend(loc=3)
   

ax1.set_xlabel('$\delta\omega/\omega_0$')
ax11.set_xlabel('$\delta\omega/\omega_0$')
ax1.set_ylabel(r'$|S_{11}|$ (dB)')
ax11.set_ylabel(r'$\angle\ S_{11}$ ($\pi$)')

axpol.set_xlabel(r'$\mathcal{Re}\ S_{11}$')
axpol.set_ylabel(r'$\mathcal{Im}\ S_{11}$')
    
plt.show()
plt.close()
```

```python
fig=plt.subplot(projection='polar')
plt.plot(np.angle(S11(ki,ki,w0,w)),np.abs(S11(ki,ki,w0,w)),label='$\kappa_e=\kappa_i$')
plt.plot(np.angle(S11(ki,ki/10,w0,w)),np.abs(S11(ki,ki/10,w0,w)),label='$\kappa_e=0.1\kappa_i$')
plt.plot(np.angle(S11(ki,ki*10,w0,w)),np.abs(S11(ki,ki*10,w0,w)),label='$\kappa_e=10\kappa_i$')
plt.legend(loc=4)
plt.show()
plt.close()
```

```python

```
