import numpy as np

def cosh(x): return 0.5*(np.exp(x)+np.exp(-x))

def weight_w(omega: np.ndarray) -> np.ndarray:
    """
    w(omega) = omega^2 * cosh(omega/2)
    """
    omega = np.asarray(omega)
    return (omega**2) * cosh(omega/2.0)

def Pn_coeffs(n: int):
    """
    Return coefficients of P_n(x) = sum_{2k<=n} binom(n,2k) * x^{k-1} / (2k-1)! for k>=1.
    Represent as list [a0, a1, ..., a_{K-1}] where x^{i} term has coefficient a_i.
    """
    from math import comb, factorial
    coeffs = []
    # k runs 1..floor(n/2); index for x^{k-1}
    for k in range(1, n//2 + 1):
        c = comb(n, 2*k) / factorial(2*k - 1)
        coeffs.append(c)
    return coeffs  # length = floor(n/2)

def Pn_eval(n: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate P_n(x) using the coeff representation.
    """
    coeffs = Pn_coeffs(n)
    if len(coeffs) == 0:
        # n=1 -> empty sum; define P1 = 0 so K_lin=0 (odd-only case)
        return np.zeros_like(x)
    y = np.zeros_like(x, dtype=np.float64)
    # y = sum_{i=0}^{K-1} coeffs[i] * x^i
    powx = np.ones_like(x, dtype=np.float64)
    for i, a in enumerate(coeffs):
        if i > 0:
            powx = powx * x
        y += a * powx
    return y

def K_lin_symbol(n: int, omega: np.ndarray) -> np.ndarray:
    """
    Effective linear symbol: K_lin_n(omega) = w(omega) * P_n(omega^2).
    """
    w = weight_w(omega)
    x = omega**2
    Pn = Pn_eval(n, x)
    return w * Pn
