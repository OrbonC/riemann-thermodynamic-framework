
---

### `src/jets.py`
```python
import numpy as np

def cosh(x): return 0.5*(np.exp(x)+np.exp(-x))
def sinh(x): return 0.5*(np.exp(x)-np.exp(-x))

def jet_multiplier(k: int, omega: np.ndarray) -> np.ndarray:
    """
    J_k(omega) from the strip-Poisson jet:
      J_{2k} = omega^{2k} * cosh(omega/2)
      J_{2k+1} = omega^{2k+1} * sinh(omega/2)
    """
    omega = np.asarray(omega)
    if k % 2 == 0:
        p = k
        return (omega**p) * cosh(omega/2.0)
    else:
        p = k
        return (omega**p) * sinh(omega/2.0)
