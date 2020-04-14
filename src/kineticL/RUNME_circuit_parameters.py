from tlineformulas import getlinepars
import numpy as np
import scipy.constants as const
eps0 = const.epsilon_0
c0 = const.c
print()

# all units in SI

# geometric parameters of DC bias cavities

# part 1: no kinetic inductance

geo = {'s': 10e-6, 'w': 6.2e-6, 't': 50e-9, 'london': 0, 'epsr': 11.9}
tlpars = getlinepars(geo)
print('Transmission line parameters without Lk:')
[print(str(key) + ':', val) for key, val in tlpars.items()]
print()

# part 2: including kinetic inductance
# 182.1e-9 for 150nm Si
# 181.9e-9 for 140nm Si
# 181.7e-9 for 130nm Si
# 181e-9 for 100nm Si
geo['london'] = 158.5e-9
# geo['t'] = 19e-9

tlpars = getlinepars(geo)
print('Transmission line parameters with Lk:')
[print(str(key) + ':', val) for key, val in tlpars.items()]
print('vph (c0)', tlpars['vph'] / c0)
print('Lk/(Lg+Lk):', tlpars['Lk'] / (tlpars['Lg'] + tlpars['Lk']))
print()

length = 6119e-6  # from end of shunt, as in Nature Communications volume 9, Article number: 4069 (2018)
print('TL physical length (um):', length / 1e-6)
print('electric length (mm):', length / (tlpars['vph'] / c0) / 1e-3)
print('Kinetic inductance (pH):', tlpars['Lk'] * length / 1e-12)
print()

# shunt capacitor
shunt = {'area': 32739e-12 * 2, 't': 60e-9, 'epsr': 7.5}
Cstray = 0  # 4e-12  # stray capacitance
Cshunt = eps0 * shunt['epsr'] * shunt['area'] / shunt['t'] / 2 + Cstray
print('Cshunt (pF):', Cshunt / 1e-12)
print()

# remember to change prefactor: 2 for shorted lambda/2, 8 for open lambda/4
Clumped = 2 * tlpars['Cg'] * length / np.pi**2
print('Clumped (pF):', Clumped / 1e-12)

print('\n########### Without kinetic inductance')
Llumped = (tlpars['Lg'] + 0) * length / 2
print('Llumped (nH):', Llumped / 1e-9)

f0unload = 1 / (2 * np.pi * np.sqrt(Llumped * Clumped))
f0load = 1 / (2 * np.pi * np.sqrt(Llumped * Clumped * Cshunt /
                                  (Clumped + Cshunt)))
print('f0unload (GHz):', f0unload / 1e9)
print('f0load (GHz):', f0load / 1e9)

Qext = 2*np.pi*f0load*tlpars['Z0'] * \
    Cshunt*(Clumped+Cshunt)/Clumped
print('Qext:', Qext)

print('\n########### With kinetic inductance')
Llumped = (tlpars['Lg'] + tlpars['Lk']) * length / 2
print('Llumped (nH):', Llumped / 1e-9)

f0unload = 1 / np.sqrt(Llumped * Clumped) / (2 * np.pi)
f0load = np.sqrt(
    (Clumped + Cshunt) / (Llumped * Clumped * Cshunt)) / (2 * np.pi)
print('f0unload (GHz):', f0unload / 1e9)
print('f0load (GHz):', f0load / 1e9)

Qext = 2*np.pi*f0load*tlpars['Z0_kinetic'] * \
    Cshunt*(Clumped+Cshunt)/Clumped
print('Qext:', Qext)
