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
def DOS(e,d):
    return abs(e)/np.sqrt(e**2-d**2)
```

```python
d=1
e0 = 3
e=np.linspace(-e0,e0,101)
```

```python
dy = 0.8
plt.fill_betweenx(e,dy,DOS(e,d)+dy)
plt.fill_betweenx(e,-DOS(e,d)-dy,-dy)
plt.xlim(-4,4)
plt.axis('off')
```

```python

```
