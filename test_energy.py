import numpy as np
from src.kernels import weight_w
from src.li_framework import gaussian_smoother, energy_E2

def mhat_gaussian(omega, sigma=1.0):
    return np.exp(-(omega**2)/(2.0*sigma**2))

def test_E2_contraction_under_gaussian_smoothing():
    omega = np.linspace(-12.0, 12.0, 4801)
    mhat = mhat_gaussian(omega, sigma=1.2)

    E = energy_E2(mhat, omega)

    for eps in [0.1, 0.5, 1.0, 2.0]:
        g = gaussian_smoother(omega, eps)
        E_eps = energy_E2(g*mhat, omega)
        assert E_eps <= E + 1e-12

def test_E2_limits_with_smoothing():
    omega = np.linspace(-12.0, 12.0, 4801)
    mhat = mhat_gaussian(omega, sigma=0.8)
    # eps -> 0+: no change
    for eps in [1e-4, 1e-3]:
        E0 = energy_E2(mhat, omega)
        E_eps = energy_E2(gaussian_smoother(omega, eps)*mhat, omega)
        assert abs(E_eps - E0) < 1e-3 * (E0 + 1.0)
    # eps large: energy decays
    for eps in [4.0, 8.0]:
        E_eps = energy_E2(gaussian_smoother(omega, eps)*mhat, omega)
        assert E_eps < 1e-3
