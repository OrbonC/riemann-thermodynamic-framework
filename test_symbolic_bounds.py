import numpy as np
import sympy as sp

from src.symbolic_bounds import (
    Pn_coeffs, Pn_poly, eval_Pn_at,
    eval_kappa_n, kappa_n_symbolic_bound
)

def test_Pn_coeffs_nonnegative_symbolic_up_to_12():
    for n in range(1, 13):
        coeffs = Pn_coeffs(n)
        # All coeffs should be >= 0 (symbolically)
        for c in coeffs:
            assert sp.simplify(c) >= 0

def test_Pn_eval_monotone_numeric():
    xs = np.linspace(0.0, 6.0, 601)
    for n in [2,3,4,5,8,12]:
        ys = eval_Pn_at(n, xs)
        # Non-decreasing on x>=0 with small numerical slack
        assert np.all(np.diff(ys) >= -1e-10)

def test_kappa_n_finite_and_decays_with_eps():
    # For a few n, kappa_n(eps) should be finite and decrease as eps increases
    ns = [2, 3, 4]
    eps_list = [0.25, 0.5, 1.0, 2.0]
    for n in ns:
        vals = []
        for e in eps_list:
            k = eval_kappa_n(n, e)
            assert np.isfinite(k), f"kappa_n({n},{e}) not finite"
            vals.append(k)
        # monotone non-increasing (allow tiny numerical slack)
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-9), f"kappa_n not decreasing for n={n}: {vals}"

def test_kappa_n_symbolic_returns_pieces_with_expected_shape():
    # Ensure the assembler returns a non-empty list of KappaPiece for n>=3
    for n in [2, 3, 4]:
        kexpr, pieces = kappa_n_symbolic_bound(n, 0.5)
        # kexpr finite
        assert np.isfinite(float(sp.N(kexpr)))
        # pieces should exist for n>=3 (n=2 has only one nonlinear term in our framework)
        if n >= 3:
            assert len(pieces) > 0
            # check fields look sane
            for p in pieces[:5]:
                assert isinstance(p.p, int) and p.p >= 0
                assert p.a >= 0.0
                assert p.b > 0.0
                assert p.C >= 0.0
