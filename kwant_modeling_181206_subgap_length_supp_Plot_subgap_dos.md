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

```python
basepath = '/home/jovyan/steelelab/projects/Felix/programming/kwant/181206_subgap_length/'
```

```python
files_dos = sorted(glob.glob(basepath + 'data/processed/*subgap_dos.pkl'))
files_evsk = sorted(glob.glob(basepath + 'data/*subgap_evskp.pkl'))

print([x.split('/')[-1] for x in files_dos])
print([x.split('/')[-1] for x in files_evsk])
```

```python
dosfile, efile = files_dos[0], files_evsk[0]
```

```python
efile
```

```python
thres = 1
```

```python
def get_e_of_k(efile):
    data1 = pickle.load(open(efile, 'rb'))
    print(data1.keys())
    all_es = data1['Energy (t)'] / data1['Delta'].values[0]
    L_N = data1['L_N'].values[0]

    #thres = 1.00  # xetex runs out of memory otherwise
    energies = all_es[abs(all_es) <= thres]
    kparallels = data1['kparallel (1/a)']
    kparallels = kparallels[abs(all_es) <= thres]
    
    LN = data1['L_N'].values[0]
    Gamma = data1['gamma_L'].values[0]
    
    return kparallels,energies,LN,Gamma,data1
```

```python
def get_dos_of_e(dosfile):
    data2 = pickle.load(open(dosfile, 'rb'))
    evals = data2['Energy (Delta)']
    dos = data2['DOS (a.u.)']
    emin = data2['Emin (Delta)'].values[0]
    return evals,dos
```

```python
id1,id2=0,1
```

```python
ek1 = get_e_of_k(files_evsk[id1]) 
ek2 = get_e_of_k(files_evsk[id2]) 
```

```python
ek1[-1].head()
```

```python
ek2[-1].head()
```

```python
doe1 = get_dos_of_e(files_dos[id1])
doe2 = get_dos_of_e(files_dos[id2])
```

```python
toy = plt.imread('plots/2_4_4_green.png')
```

```python
fig = plt.figure(figsize=cm2inch(17, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 2, wspace=0.3)
#ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

#ax0.imshow(toy)
#ax0.axis('off')

ax1.scatter(ek1[0], ek1[1], marker='.', s=0.4, label='long')#ek1[2])
ax1.scatter(ek2[0], ek2[1], marker='.', s=0.4, label='short')#ek2[2])
ax1.set_ylim(-thres, thres)
ax1.set_xlim(0, 0.3)
ax1.set_xlabel(r'$k_{\parallel}$ ($1/a$)')
ax1.set_ylabel(r'$E$ ($\Delta$)')
ax1.axhline(1, c="k", linestyle='--', lw=1)
ax1.axhline(-1, c="k", linestyle='--', lw=1)
#ax1.axhline(min(abs(energies)), c="k", linestyle='--',lw=1)
#ax1.axhline(-min(abs(energies)), c="k", linestyle='--',lw=1)
ax1.text(-0.2,
         1.,
         r'(a)',
         transform=ax1.transAxes,
         verticalalignment='center',
         horizontalalignment='center',
         weight='bold')

#ax2.plot(doe1[0], doe1[1])#, label='long')
#ax2.plot(doe2[0], doe2[1])#, label='short')
ax2.fill_between(doe1[0], 0, doe1[1],label='long',alpha=0.8,edgecolor='C0')
ax2.fill_between(doe2[0], 0, doe2[1], zorder=-99,label='short',alpha=0.8,edgecolor='C1')
ax2.set_xlim(-thres, thres)
ax2.set_ylim(0.0, 0.4)
ax2.set_xlabel(r'$E$ ($\Delta$)')
ax2.set_ylabel(r'DOS (a.u.)')
#ax2.axvline(1, c="k", linestyle='--',lw=1)
#ax2.axvline(-1, c="k", linestyle='--',lw=1)
#ax2.axvline(emin, c="k", linestyle='--', lw=1)
#ax2.axvline(-emin, c="k", linestyle='--', lw=1)
ax2.text(-0.2,
         1.,
         r'(b)',
         transform=ax2.transAxes,
         verticalalignment='center',
         horizontalalignment='center',
         weight='bold')

ax1.legend(markerscale=10)
ax2.legend()

plt.savefig('plots/kwant_modeling_181206_subgap_length_supp_Plot_subgap_dos.pdf',bbox_to_inches='tight',dpi=dpi)
plt.show()
plt.close()
```

```python

```
