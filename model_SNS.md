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

# Model SNS characteristics

```python
%run src/basemodules.py
```

## DOS

```python
def DOS(e, d):
    return abs(e) / np.sqrt(e**2 - d**2)
```

```python
d = 1
e0 = 3
elow = np.linspace(-e0, d, 101)
ehigh = np.linspace(d, e0, 101)
```

```python
def circle(y,r):
    return np.sqrt(r**2-y**2)
```

```python
from matplotlib.text import Annotation
```

```python hide_input=false
fig = plt.figure(figsize=cm2inch(17,6))#,constrained_layout=True)
gs = fig.add_gridspec(1,2,width_ratios=[3,4])#,wspace=10)

# SIS
dy = 0.2
xe = -0.2
xh = -xe
xer = dy + 1
xhl = -xer - 1
ye = 0.5
yh = -ye
yel, yhl, yhr, yer = 0, 0, 0, 0
dx = 0.25
xhr = xer + dx
xel = xhl - dx
da = 0.25
yABS = np.linspace(yel-da,yel+da,101)
ycirc = np.linspace(yh,ye,101)

ax1 = fig.add_subplot(gs[0,0])
# DOS
plt.fill_betweenx(elow, dy, DOS(elow, d) + dy, facecolor='C0', edgecolor='k')
plt.fill_betweenx(ehigh, dy, DOS(ehigh, d) + dy, facecolor='none', edgecolor='k')
plt.fill_betweenx(elow, -DOS(elow, d) - dy, -dy, facecolor='C0',edgecolor='k')
plt.fill_betweenx(ehigh, -DOS(ehigh, d) - dy, -dy, facecolor='none', edgecolor='k')
plt.axvline(-dy, c='k', ls='-')
plt.axvline(dy, c='k', ls='-')

# Cooper pairs
plt.scatter([xel, xer], [yel, yer],s=30)
plt.scatter([xhl, xhr], [yhl, yhr],c='C0',s=30)

# Cooper pairs
plt.plot(circle(yABS,ye/2)+xhl,yABS,'--',c='grey') # left left 
plt.plot(-circle(yABS,ye/2)+xel,yABS,'--',c='grey') # left right 
plt.plot(-circle(yABS,ye/2)+xer,yABS,'--',c='grey') # right left 
plt.plot(circle(yABS,ye/2)+xhr,yABS,'--',c='grey') # right right 

# arrows
ax1.arrow(xhr+0.5, yhr, 0.5, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')
ax1.arrow(xhl+0.5, yel, 0.5, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')
ax1.arrow(-1.5*dy, 0, 3*dy, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')

# misc
plt.xlim(-3, 3)
plt.axis('off')


# SNS

dy = 0.8
xe = -0.2
xh = -xe
xer = dy + 1
xhl = -xer - 1
ye = 0.5
yh = -ye
yel, yhl, yhr, yer = 0, 0, 0, 0
dx = 0.25
xhr = xer + dx
xel = xhl - dx
da = 0.25
yABS = np.linspace(yel-da,yel+da,101)
ycirc = np.linspace(yh,ye,101)

ax2 = fig.add_subplot(gs[0,1])

# DOS
plt.fill_betweenx(elow, dy, DOS(elow, d) + dy, facecolor='C0', edgecolor='k')
plt.fill_betweenx(ehigh, dy, DOS(ehigh, d) + dy, facecolor='none', edgecolor='k')
plt.fill_betweenx(elow, -DOS(elow, d) - dy, -dy, facecolor='C0',edgecolor='k')
plt.fill_betweenx(ehigh, -DOS(ehigh, d) - dy, -dy, facecolor='none', edgecolor='k')
plt.fill_between([-dy,dy],[-3,-3],[0,0],facecolor='k',edgecolor='k',alpha=0.3)
plt.axvline(-dy, c='k', ls='-')
plt.axvline(dy, c='k', ls='-')

# Cooper pairs
plt.scatter([xe, xel, xer], [ye, yel, yer],s=30)
plt.scatter([xhl, xhr], [yhl, yhr],c='C0',s=30)

# ABS
plt.scatter([xe], [ye],c='C0',s=30)
plt.scatter([xh], [yh],c='none',edgecolors='C0',s=30)

# paths N
plt.plot([-dy, dy], [ye, ye], '-')
plt.plot([-dy, dy], [yh, yh], 'C0--')

# paths S
plt.plot(circle(ycirc,ye)+dy,ycirc,'--',c='grey')
plt.plot(-circle(ycirc,ye)-dy,ycirc,'--',c='grey')

# Cooper pairs
plt.plot(circle(yABS,ye/2)+xhl,yABS,'--',c='grey') # left left 
plt.plot(-circle(yABS,ye/2)+xel,yABS,'--',c='grey') # left right 
plt.plot(-circle(yABS,ye/2)+xer,yABS,'--',c='grey') # right left 
plt.plot(circle(yABS,ye/2)+xhr,yABS,'--',c='grey') # right right 

# arrows
ax2.arrow(xe, ye, 0.5, 0, head_width=0.2, head_length=0.2, fc='C0', ec='C0')
ax2.arrow(xh, yh, -0.5, 0, head_width=0.2, head_length=0.2, fc='C0', ec='C0')
ax2.arrow(xhr+0.5, yhr, 0.5, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')
ax2.arrow(xhl+0.5, yel, 0.5, 0, head_width=0.2, head_length=0.1, fc='k', ec='k')

# misc settings
plt.xlim(-4, 4)
plt.axis('off')

## Fermi level
for theax in [ax1,ax2]:
    plt.sca(theax)
    plt.axhline(0,c='k',ls='--',alpha=0.5,zorder=-99)

## JJ text
ax1.text(-dy-1,-3.5,'S',weight='bold',ha='right')
ax1.text(dy+1,-3.5,'S',weight='bold',ha='left')
ax1.text(0,-3.5,'I',weight='bold',ha='center')
ax2.text(-dy-1.5,-3.5,'S',weight='bold',ha='right')
ax2.text(dy+1.5,-3.5,'S',weight='bold',ha='left')
ax2.text(0,-3.5,'N',weight='bold',ha='center')

## misc text
#ax1.text(3.3,0,r'$\epsilon_{\rm F}$',weight='bold',ha='center',va='center')
#ax1.arrow(-e0, -d, 0, 2*d, head_width=0.2, head_length=0.2, fc='k', ec='k') # this will take some extra work for the arrow not to be out of bounds of the plot...

ax1.text(0, .95, "(a)", weight="bold", transform=ax1.transAxes)
ax2.text(0, .95, "(b)", weight="bold", transform=ax2.transAxes)

plt.savefig('plots/model_SNS_DOS.pdf',bbox_to_inches='tight',dpi=dpi)
plt.show()
plt.close()
```

## Ej and Ic

```python
def Ej(phi,tau,tau_c=1):
    # tau_c: transparency at SN interface
    # tau: transmission in the normal region
    # phi: phase
    return tau_c*np.sqrt(1 - tau * np.sin(phi / 2)**2)
    
def Ic(phi,tau):
    return 1/2*tau*np.sin(phi)/Ej(phi,tau)

def Lj(phi,tau):
    return 2*Ej(phi,tau)**3/(tau*np.cos(phi)*Ej(phi,tau)**2+1/4*tau**2*np.sin(phi)**2)

def phimax_of_I(tau):
    phi=np.linspace(0,2*pi,10001)
    Ij = Ic(phi,tau)
    return phi[np.argmax(Ij)]
```

```python
phi=np.linspace(0,2*pi,101)
```

```python
taus = [0.3,0.5,0.7,0.9]
```

```python
tc=0.9
```

```python
fig = plt.figure(figsize=cm2inch(17, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 3)

ax1 = fig.add_subplot(gs[0, 0])
for tau in taus:
    plt.plot(phi/pi, Ej(phi, tau,tc), 'C1', alpha=tau + .1)
    plt.plot(phi/pi, -Ej(phi, tau,tc), 'C0', alpha=tau + .1,label=tau)
#plt.plot(phi/pi, Ej(phi, 1), '--k', alpha=0.7)
#plt.plot(phi/pi, -Ej(phi, 1), '--k', alpha=0.7)
plt.axhline(1,c='k',ls='--',alpha=0.7)
plt.axhline(-1,c='k',ls='--',alpha=0.7)
plt.legend()

ax2 = fig.add_subplot(gs[0, 1])
for tau in taus:
    plt.plot(phi/pi, Ic(phi, tau), 'C0', alpha=tau + .1,label=tau)
plt.legend()

ax3 = fig.add_subplot(gs[0, 2])
for tau in taus:
    phi2 = np.linspace(0,phimax_of_I(tau)-0.01,len(phi))
    plt.plot(phi2/phimax_of_I(tau), Lj(phi2, tau), 'C0', alpha=tau + .1,label=tau)
plt.legend()

for theax in [ax1,ax2]:
    theax.set_xlabel(r'$\delta$ ($\pi$)')

ax1.set_ylabel(r'$U_J$ $(\Delta)$')

ax2.set_ylabel(r'$I_J$ $(\frac{e\Delta}{\hbar})$')

ax3.set_ylabel(r'$L_J$ $(\frac{4\hbar^2}{e^2\Delta})$')
ax3.set_ylim(0,20)
ax3.set_xlabel(r'$\delta$ ($\delta_{\rm max}$)')

ax1.text(-0.45, .95, "(a)", weight="bold", transform=ax1.transAxes)
ax2.text(-0.5, .95, "(b)", weight="bold", transform=ax2.transAxes)
ax3.text(-0.4, .95, "(c)", weight="bold", transform=ax3.transAxes)

plt.savefig('plots/model_SNS_EjIc.pdf',bbox_to_inches='tight',dpi=dpi)
plt.show()
plt.close()
```

```python
for tau in taus:
    plt.plot(1/np.gradient(Ic(phi,tau),phi))
    plt.plot(Lj(phi,tau),'--')
plt.ylim(-10,10)
```

```python

```
