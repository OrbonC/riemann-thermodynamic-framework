import numpy as np
from src.li_framework import positivity_margin, lambda_n_linear, energy_E2, gaussian_smoother

def mhat_gaussian(omega, sigma=1.0):
    return np.exp(-(omega**2)/(2.0*sigma**2))

def mhat_bump(omega):
    return np.exp(-omega**2) * (1.0 + 0.1*np.cos(3*omega))

def _grid():
    return np.linspace(-12.0, 12.0, 4801)

def test_linear_part_positive_for_n2_n3_various_mhats():
    omega = _grid()
    for mfun in [lambda w: mhat_gaussian(w,1.0),
                 lambda w: mhat_gaussian(w,0.6),
                 mhat_bump]:
        mhat = mfun(omega)
        L2 = lambda_n_linear(2, mhat, omega)
        L3 = lambda_n_linear(3, mhat, omega)
        assert L2 >= -1e-10
        assert L3 >= -1e-10

def test_margin_lower_bound_nontrivial_with_large_eps():
    omega = _grid()
    mhat = mhat_gaussian(omega, sigma=1.0)

    # choose eps large so that E2[m_eps] is tiny -> lower bound likely positive
    eps = 2.0

    out2 = positivity_margin(2, mhat, omega, eps=eps, delta=0.2)
    out3 = positivity_margin(3, mhat, omega, eps=eps, delta=0.2)

    # Internally consistent:
    assert out2["kappa"] is not None and out3["kappa"] is not None
    assert out2["E_eps"] <= energy_E2(mhat, omega) + 1e-12
    assert out3["E_eps"] <= energy_E2(mhat, omega) + 1e-12

    # Lower bounds should be finite; with eps=2.0 typically positive on our test mhat.
    assert np.isfinite(out2["lower_bound"])
    assert np.isfinite(out3["lower_bound"])
    assert out2["lower_bound"] > -1e-8
    assert out3["lower_bound"] > -1e-8
