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

# Current-biasing a Josephson junction


\begin{align}
U(\delta) = -E_J \left( \cos\delta + \frac{I_b}{I_c}\delta \right)
\end{align}

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('white')
sns.set_style('ticks')
```

```python
from numpy import pi,cos,sin
```

```python
delta = np.linspace(-2*pi,8*pi,401)
```

```python
currs = [0,0.5,1,1.5]
```

```python
from matplotlib import colors
```

```python
# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
```

## sinusoidal CPR

```python
def U(delta,i):
    return 1-(cos(delta)+i*delta)
```

```python
plt.plot(delta,U(delta,0))
```

```python
# here we calculate the location of the phase particle under the different conditions
# the particle will start at phi=0 and come to rest (if at all) at the location where the total current is zero.
```

```python
from scipy.optimize import brentq, root
from scipy.interpolate import interp1d
```

```python
def zero_current(delta,i):
    phi = np.linspace(-10*pi,10*pi,1001)
    return interp1d(phi,np.gradient(U(phi,i),phi))(delta)
```

```python
phi = np.linspace(-pi,pi*2,401)
```

```python
ii = np.linspace(0,0.99,21)
#ii
```

```python
sol = np.array([brentq(zero_current,-0.5*pi,0.5*pi,args=(i,)) for i in ii])
```

```python
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
plt.sca(ax1)
for i in currs:
    plt.plot(delta/pi,U(delta,i)-i*10,label=i)
plt.plot(sol[0],U(sol[0],currs[0])-currs[0]*10,'ok')
plt.plot(sol[1],U(sol[1],currs[1])-currs[1]*10,'ok')
plt.legend()
plt.sca(ax2)
for i in currs:
    plt.plot(delta/pi,np.gradient(U(delta,i),delta),label=i)
plt.legend()
plt.show()
plt.close()
```

```python
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
plt.sca(ax1)
for i,s in zip(ii,sol):
    plt.plot(phi/pi,U(phi,i)-i*10,label=i,c='k',ls='--')
    plt.plot(s/pi,U(s,i)-i*10,'ok')
    
plt.sca(ax2)
plt.plot(ii,sol/pi,'ok')
```

```python
X,Y = np.meshgrid(delta,np.linspace(0,1.5,len(delta)))
Z = U(X,Y)
Z2 = np.gradient(U(X,Y))[1]
```

```python
#pcm = plt.pcolormesh(X/pi,Y,Z)#,vmin=-0.1,vmax=0.1,cmap='PiYG')
pcm = plt.pcolormesh(X/pi,Y,Z2,cmap='PiYG',norm=MidpointNormalize(vmin=np.min(Z2), midpoint=0, vmax=np.max(Z2)))
plt.colorbar()
plt.contour(X/pi,Y,Z2,levels=[0],colors='k',linestyles='--')
plt.show()
plt.close()
```

## short ballistic

```python
def U2_2D(delta,i,tau,**kwargs):
    x1 = np.sqrt(1-tau*sin(delta/2)**2)
    dx = (np.nanmax(x1)+np.nanmin(x1))/2
    x2 = x1-dx
    dy = (np.nanmax(x2)-np.nanmin(x2))/2
    x = 1-x2/dy-i*delta
    return x

def U2(delta,i,tau,**kwargs):
    phi = np.linspace(-10*pi,10*pi,1001)
    x1 = np.sqrt(1-tau*sin(phi/2)**2)
    dx = (np.nanmax(x1)+np.nanmin(x1))/2
    x2 = x1-dx
    dy = (np.nanmax(x2)-np.nanmin(x2))/2
    x = 1-x2/dy-i*phi
    return interp1d(phi,x,fill_value='extrapolate')(delta)
```

```python
mytau=1-1e-5
```

```python
i=0
plt.plot(delta/pi,U2(delta,i,tau=0.01),label=i)
plt.plot(delta/pi,U2(delta,i,tau=0.99),label=i)
```

```python
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
plt.sca(ax1)
for i in currs:
    plt.plot(delta/pi,U2(delta,i,tau=mytau),label=i)
plt.sca(ax2)
for i in currs:
    plt.plot(delta/pi,np.gradient(U2(delta,i,tau=mytau),delta),label=i)
plt.show()
plt.close()
```

```python
def zero_current_tau(delta,i,tau):
    phi = np.linspace(-10*pi,10*pi,1001)
    return interp1d(phi,np.gradient(U2(phi,i,tau),phi))(delta)
```

```python
ii = np.linspace(0,0.99,21)
sol = np.array([brentq(zero_current,-0.5*pi,0.5*pi,args=(i,)) for i in ii])
sol_tau = np.array([brentq(zero_current_tau,-0.1*pi,0.9*pi,args=(i,mytau)) for i in ii])
```

```python
fig=plt.figure(figsize=(8,3.5),constrained_layout=True)
gs=fig.add_gridspec(2,2)
ax2=fig.add_subplot(gs[1,0])
for i,st in zip(ii,sol_tau):
    plt.plot(phi/pi,U2(phi,i,mytau)-i*10,label=i,c='C1',ls='-')
    plt.plot(st/pi,U2(st,i,mytau)-i*10,'.C1')
plt.text(-1,-13, f'{mytau:.1f}', ha = 'left',va='top')
ax1=fig.add_subplot(gs[0,0])
for i,s in zip(ii,sol):
    plt.plot(phi/pi,U(phi,i)-i*10,label=i,c='C0',ls='-')
    plt.plot(s/pi,U(s,i)-i*10,'.C0',label='SIS')
plt.text(-1,-13, 'SIS', ha = 'left',va='top')
ax3=fig.add_subplot(gs[:,1])
plt.plot(ii,sol_tau/pi,'oC1',label=f'{mytau:.1f}')
plt.plot(ii,sol/pi,'oC0',label='SIS')
myphi = np.linspace(-1.01*pi,1.01*pi,101)
plt.plot(np.gradient(U2(myphi,0,tau=mytau),myphi),myphi/pi,'C1')
plt.plot(np.gradient(U(myphi,0),myphi),myphi/pi,'C0')
plt.legend()

ax1.set_xticklabels([])
ax2.set_xlabel(r'Phase ($\pi$)')
ax3.set_ylabel(r'Phase minimum ($\pi$)')
ax3.set_xlabel(r'Bias current ($I_c$)')
[theax.set_ylabel(r'$E_J$ (a.u.)') for theax in [ax1,ax2]]
#plt.suptitle('Bias current dependence')
plt.show()
plt.close()
```

**Bias current dependence of Josephson junctions and their energy potential.
(a,b)** Josephson energy potential for an ideal SIS JJ **(a)** and a ballistic SNS one with high transmission **(b)**.
With increasing bias current, the energy potential tilts and the phase particle moves forward.
**(c)** The location of the phase particle as a function of bias current maps out the current-phase relation of the respective JJs.

```python

```
