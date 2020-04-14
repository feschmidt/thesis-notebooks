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

# Modelling the Josephson energy landscape


References:
* Kringhøj _et al._ , Phys. Rev. B **97**, 060508(R) (2018) [_Anharmonicity of a superconducting qubit with a few-mode Josephson junction_](https://doi.org/10.1103/PhysRevB.97.060508)
* titovJosephsonEffectBallistic2006b. From here it is also expected for the supercurrent to always be present, even at zero charge carrier density. At the CNP, the CPR should be similar to the one of a diffusive SNS contact (i.e. almost sinusoidal). This goes back to the quantum-limited minimum conductivity $e^2/h$ in graphene, see refs. katsnelsonZitterbewegungChiralityMinimal2006, zieglerRobustTransportProperties2006, tworzydloSubPoissonianShotNoise2006.
* more papers on anharmonicity in nanowire transmons: larsenSemiconductorNanowireBasedSuperconductingQubit2015, delangeRealizationMicrowaveQuantum2015, casparisGatemonBenchmarkingTwoQubit2016acasparisSuperconductingGatemonQubit2017

\begin{align}
\frac{\partial\delta}{\partial t}&=\frac{2eV_J}{\hbar}\, ,\, V_J=L_J\frac{\partial I_J}{\partial t}=L_J\frac{\partial I_J}{\partial\delta}\frac{\partial\delta}{\partial t}
\end{align}

The work done on the junction must be $U_J=\int I_J V_J {\rm d}t=\frac{\hbar}{2e}\int I_J \frac{\partial\delta}{\partial t}{\rm d}t = \frac{\hbar}{2e}\int I_J {\rm d}\delta$, therefore $I(\delta) = \frac{2e}{\hbar}\frac{\partial U_J}{\partial\delta}$.
Each Andreev bound state has a ground state energy $-\Delta\sqrt{1-T_i\sin^2(\delta/2)}$ with induced superconducting gap $\Delta$ and channel transparency $T_i$.
Summing over all of these channels, the total Josephson potential is given by
\begin{align}
U_J(\delta) &= 1-\Delta\sum_i\sqrt{1-T_i\sin^2(\delta/2)} \\
 &\approx E_J \frac{\delta^2}{2} - E_J\left( 1-\frac{3\sum T_i^2}{4\sum T_i} \right) \frac{\delta^4}{24} +\mathcal{O}(\delta^6)\ , \\
 E_J &= \frac{\Delta}{4}\sum_i T_i \rightarrow \frac{\Delta}{4}N\tau
\end{align}

The corresponding Josephson current and inductance are 
\begin{align}
I(\delta) &= \frac{2e}{\hbar}\frac{\partial U_J}{\partial\delta} = \frac{e\Delta}{2\hbar}\frac{\tau\sin\delta}{\sqrt{1-\tau\sin^2\delta/2}} \\
 L_J(\delta) &= \frac{\hbar}{2e}\left( \frac{\partial I_J}{\partial\delta} \right)^{-1} = \left(\frac{\hbar}{2e}\right)^2\left(\frac{\partial^2U_J}{\partial\delta^2}\right)^{-1}
\end{align}

The anharmonicity in the circuit is then reduced depending on the channel transparencies.
In the limit of small $L_J$, we can model the TL resonator as a series $LC$-circuit, resulting in anharmonicity
\begin{align}
\chi &= \frac{-E_C}{2} \left( 1-\frac{3\sum T_i^2}{4\sum T_i} \right) p^3 \rightarrow \frac{-E_C}{2} \left(1-\frac{3}{4}N\tau \right) p^3 
\end{align}

Here, $\chi$ has units of energy, $p=L_J/(L_r+L_J)$ is the participation ratio of the Josephson to total circuit inductance, and $E_C=e^2/(2C)=\hbar\omega_0$ the charging energy of the circuit capacitance.
Depending on notation, $\chi$ may or may not include factors of $\hbar,2\pi,12$ or $24$.
$L_J$ itself however also includes a transparency.

We should therefore be able to extract a value for the junction transparency by dividing the measured anharmonicity by the calculated one, resulting in $\tau=\frac{4}{3N}\left(1-\frac{\chi_{\rm meas}}{\chi_{\rm calc}}\right)\in[0,1]$ and $\frac{1}{4}\leq\frac{\chi_{\rm meas}}{\chi_{\rm calc}}\leq1$.


One additional thing which we can see from the CPR, but also in literature, the skewness (and therefore the transparency) is expected to be higher for larger values of Ic.
references:
* Manjarrés _et al._ , Phys. Rev. B **101**, 064503 (2020) [_Skewness and critical current behavior in a graphene Josephson junction_](https://doi.org/10.1103/PhysRevB.101.064503)
* Lee and Lee, Rep. Prog. Phys. **81** (2018) 056502 [_Proximity coupling in superconductor-graphene heterostructures_](https://doi.org/10.1088/1361-6633/aaafe1)

This is actually reverse; the higher the transparency, the higher the probability that ABS pairs get transferred across the normal region multiple times, leading to higher order frequencies and resulting in forward skew.
This higher transparency consequently increases the critical current.
Note that this consideration only holds for constant chemical doping.
However, since we can assume symmetric carrier density as a function of applied gate voltage due to the linear band dispersion of graphene, the junction should exhibit higher $\tau$ and $S$ for n- compared to p-doping.

```python
%load_ext autoreload
%autoreload 2
```

```python
%run src/basemodules.py
```

```python
delta = np.linspace(0,pi,401)
```

## CPR and temp

```python
plt.plot(delta/pi,np.sin(delta),'k--',label=r'$\sin(\delta)$')
plt.plot(delta/pi,CPR(delta,tau=0.9,T=1e-3,norm=True,Delta=.18e-3),label='1 mK')
plt.plot(delta/pi,CPR(delta,tau=0.9,T=1,norm=True,Delta=.18e-3),label='1 K')
plt.legend()
plt.xlabel('Phase ($\pi$)')
plt.ylabel('Josephson current')
plt.title('Effect of temperature')
```

## EJ and tau

```python
def E_ABS(phase, tau):
    if tau == 0:
        return 1 - np.cos(phase)
    else:
        return 1 - np.sqrt(1 - tau * np.sin(phase / 2)**2)
```

```python
def EJ(tau, Delta=1):
    if tau == 0:
        return Delta
    else:
        return Delta * tau / 4
```

```python
def VJ(phi, tau, Delta=1, norm=True):
    if norm:
        ej = Delta
    else:
        ej = Delta / 4 * tau
    if tau > 0:
        return ej * phi**2 / 2 - ej * (1 - 3 / 4 * tau) * phi**4 / 24
    else:
        return 1 - np.cos(phi)
```

```python
phi = np.linspace(-1.1*pi,1.1*pi,401)
```

```python
plt.plot(phi/pi,(VJ(phi,0)))
plt.plot(phi/pi,(VJ(phi,1)))
plt.plot(phi/pi,phi**2/2)
plt.ylim(0,5)
#plt.xlim(-1,1)
```

```python
for tau in [0,0.5,1]:
    plt.plot(phi/pi,E_ABS(phi,tau)/EJ(tau),label=f'{tau:.1f}')
plt.plot(phi/pi,phi**2/2,'k--',label='SIS')
plt.gca().set_ylim(top=4.5)
plt.legend()
plt.xlabel('Phase ($\pi$)')
plt.ylabel('Josephson energy ($E_J$)')
plt.title('Effect of transmission')
```

```python
for tau in [0,0.5,0.8,1]:
    plt.plot(phi/pi,2*np.gradient(E_ABS(phi,tau)/EJ(tau),phi)/max(np.gradient(E_ABS(phi,tau)/EJ(tau),phi)),'o',label=f'{tau:.1f}')
for tau in [0,0.5,0.8,1]:
    plt.plot(phi/pi,2*CPR(phi,tau),'--')
#plt.plot(phi/pi,phi,'k--',label='SIS')
#plt.gca().set_ylim(top=4.5)
plt.legend()
plt.xlabel('Phase ($\pi$)')
plt.ylabel('Josephson current ($I_J$)')
plt.title('Effect of transmission')
```

Dashed line: analytical CPR, dots: numerical derivative of ABS energy.
This shows that the analytical equation for the CPR indeed follows directly from the Andreev bound state energy.


## EJ and bias current

```python

```
