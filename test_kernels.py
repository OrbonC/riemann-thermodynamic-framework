import numpy as np
from math import isfinite

from src.kernels import Pn_coeffs, Pn_eval, K_lin_symbol, weight_w

def test_Pn_coeffs_nonnegative_up_to_20():
    for n in range(1, 21):
        coeffs = Pn_coeffs(n)
        assert all(c >= 0 for c in coeffs)

def test_Pn_eval_monotone_in_x_for_sample_n():
    # P_n(x) has nonnegative coefficients => nondecreasing on x>=0
    x = np.linspace(0.0, 9.0, 1001)
    for n in [2,3,4,5,10]:
        y = Pn_eval(n, x)
        assert np.all(np.diff(y) >= -1e-12)  # allow tiny FP noise

def test_Klin_nonnegative_on_grid():
    omega = np.linspace(-12.0, 12.0, 4801)
    for n in [2,3,4,5,8,10]:
        Klin = K_lin_symbol(n, omega)
        assert np.all(Klin >= -1e-10)  # numerical slack

def test_weight_positive_away_from_zero_and_finite_everywhere():
    omega = np.linspace(-10.0, 10.0, 2001)
    w = weight_w(omega)
    # strictly >0 except exactly at 0 where omega^2 factor gives 0
    assert np.all(w[omega != 0.0] > 0.0)
    assert all(isfinite(float(val)) for val in w)
