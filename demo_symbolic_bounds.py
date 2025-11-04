import numpy as np
from pprint import pprint
import sympy as sp

from src.symbolic_bounds import (
    Pn_coeffs, Pn_poly, eval_Pn_at, eval_kappa_n, kappa_n_symbolic_bound
)

def main():
    print("=== Linear kernel polynomials P_n(x) (symbolic) ===")
    for n in [2,3,4,5,8]:
        coeffs = Pn_coeffs(n)
        poly = Pn_poly(n)
        print(f"n={n}  coeffs={[sp.N(c) for c in coeffs]}")
        print(f"P_{n}(x) = {sp.simplify(poly)}\n")

    print("=== Sample numeric evaluation of P_n(x) ===")
    xs = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    for n in [2,3,4,5]:
        print(f"n={n}, P_n(xs) = {eval_Pn_at(n, xs)}")

    print("\n=== Conservative κ_n(ε) bounds (symbolic assembly) ===")
    for n in [2,3,4]:
        for eps_val in [0.25, 0.5, 1.0]:
            kappa_expr, pieces = kappa_n_symbolic_bound(n, eps_val)
            print(f"n={n}, eps={eps_val}: kappa_expr ~ {sp.N(kappa_expr)}")
            print("  pieces (first 5 shown):")
            pprint(pieces[:5])
        print("")

if __name__ == "__main__":
    main()
