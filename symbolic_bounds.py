"""
symbolic_bounds.py
------------------
Symbolic scaffolding for nonlinear (Faà di Bruno / Bell polynomial) terms
appearing in derivatives of log ξ(s) at s=1, and for constructing explicit,
auditable bounds κ_n(ε).

Goals:
  1) Represent S_k := ∂_s^k log ξ(s) at s=1 in terms of jets J_k(ω).
  2) Build the Faà di Bruno combinatorics via complete Bell polynomials.
  3) Produce conservative, explicit κ_n(ε) bounds so that:
         |R_n[m_ε]| ≤ κ_n(ε) · E_2[m_ε]
     with κ_n(ε) finite and *traceable* from symbolic expressions.

This file is a scaffold:
  - It makes the structure explicit and testable.
  - It leaves placeholders (`TODO`) where analytic constants must be tightened.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import sympy as sp

# -----------------------------------------------------------------------------
# Core symbolic objects
# -----------------------------------------------------------------------------

# Symbols
omega, eps = sp.symbols('omega eps', positive=True)
x = sp.symbols('x', nonnegative=True)   # for P_n(x) polynomial variable

# Useful analytic envelopes (upper bounds) for jets on real ω:
# |sinh(ω/2)| ≤ exp(|ω|/2), |cosh(ω/2)| ≤ exp(|ω|/2)
sinh_env = sp.exp(sp.Abs(omega)/2)
cosh_env = sp.exp(sp.Abs(omega)/2)

def jet_even_env(k: int) -> sp.Expr:
    """
    Envelope for even jet J_{2k}(ω) ~ ω^{2k} cosh(ω/2).
    Returns a symbolic upper bound valid for all ω ∈ ℝ.
    """
    assert k >= 1
    return (sp.Abs(omega)**(2*k)) * cosh_env

def jet_odd_env(k: int) -> sp.Expr:
    """
    Envelope for odd jet J_{2k+1}(ω) ~ ω^{2k+1} sinh(ω/2).
    """
    assert k >= 0
    return (sp.Abs(omega)**(2*k+1)) * sinh_env

# -----------------------------------------------------------------------------
# Bell polynomials & Faà di Bruno machinery
# -----------------------------------------------------------------------------

def complete_bell_polynomial(n: int, y: List[sp.Symbol]) -> sp.Expr:
    """
    SymPy wrapper for the complete Bell polynomial B_n(y1,...,yn).
    """
    # sympy.bell_polynomial takes (n, *ys) for partials; for complete use bell_polynomial(n, y1, ..., yn) with m=n
    return sp.bell_polynomial(*y[:n], n=n)

def faa_di_bruno_terms(n: int) -> List[Tuple[int, Tuple[int, ...], int]]:
    """
    Enumerate Faà di Bruno decompositions for the nth derivative of a composite:
      d^n/ds^n f(g(s)) = sum over partitions
    We represent each term by a tuple:
       (multiplier, multiplicities, r)
    where:
      multiplicities = (m1, m2, ..., mn) with sum j m_j = n and r = sum m_j
      multiplier = n! * f^{(r)}(g) * Π_j (1/m_j!) * (g^{(j)}/j!)^{m_j}
    Here we return only the combinatorial piece (integer multiplier and multiset).
    """
    # This uses the standard integer partition of n into j*m_j with multiplicities m_j.
    results = []
    n_fact = sp.factorial(n)
    # generate all tuples (m1,...,mn) with sum j*m_j = n
    # For n up to ~10 this brute force is fine; collaborators can optimize later.
    def rec(j, remaining, current):
        if j > n:
            if remaining == 0:
                m = tuple(current)
                r = sum(current)
                # multiplier: n! * Π_j 1/m_j! * 1/(j!)^{m_j}
                mult = n_fact
                for jj, mj in enumerate(m, start=1):
                    if mj:
                        mult //= sp.factorial(mj)
                        mult //= sp.factorial(jj)**mj
                results.append((int(mult), m, r))
            return
        for mj in range(0, remaining//j + 1):
            current.append(mj)
            rec(j+1, remaining - j*mj, current)
            current.pop()
    rec(1, n, [])
    return results

# -----------------------------------------------------------------------------
# Linear kernel polynomial P_n(x)
# -----------------------------------------------------------------------------

def Pn_coeffs(n: int) -> List[sp.Rational]:
    """
    Coefficients of P_n(x) = sum_{2k≤n} binom(n,2k) x^{k-1} / (2k-1)!  for k≥1.
    Returned as [a0, a1, ..., a_{K-1}] where a_i ≥ 0.
    """
    coeffs = []
    for k in range(1, n//2 + 1):
        a = sp.binomial(n, 2*k) / sp.factorial(2*k - 1)
        coeffs.append(sp.simplify(a))
    return coeffs

def Pn_poly(n: int) -> sp.Expr:
    coeffs = Pn_coeffs(n)
    poly = 0
    for i, a in enumerate(coeffs):
        poly += a * x**i
    return sp.simplify(poly)

# -----------------------------------------------------------------------------
# Gaussian smoothing envelope in Fourier space: exp(-eps * ω^2)
# -----------------------------------------------------------------------------

def gaussian_env(omega_sym: sp.Symbol, eps_sym: sp.Symbol) -> sp.Expr:
    return sp.exp(-eps_sym * omega_sym**2)

# -----------------------------------------------------------------------------
# κ_n(ε) conservative bound assembler (symbolic)
# -----------------------------------------------------------------------------

@dataclass
class KappaPiece:
    """
    A single nonlinear piece bound, of the generic form:
      C * sup_{ω≥0} ω^p * exp(a*ω - b*ε*ω^2)
    with explicit (p, a, b, C).
    """
    p: int
    a: float
    b: float
    C: float

def sup_poly_exp(p: int, a: float, b_eps: float) -> sp.Expr:
    """
    Closed-form supremum of ω^p * exp(a ω - b_eps ω^2) over ω≥0.
    Uses the standard trick: maximize log f = p log ω + a ω - b_eps ω^2.
    We deliver a conservative analytic upper bound using SymPy for critical point;
    if it fails, fall back to numeric.
    """
    w = sp.symbols('w', nonnegative=True)
    f = (w**p) * sp.exp(a*w - b_eps*w**2)
    # derivative and solve
    df = sp.diff(sp.log(f), w)
    crit = sp.solve(sp.Eq(df, 0), w, domain=sp.Interval(0, sp.oo))
    val_candidates = []
    for sol in crit:
        try:
            if sol.is_real and sol >= 0:
                val_candidates.append(sp.simplify(f.subs(w, sol)))
        except Exception:
            pass
    # boundary at 0
    val_candidates.append(sp.simplify(f.subs(w, 0)))
    # if no symbolic solution, do numeric probe
    if not val_candidates:
        grid = np.linspace(0.0, float(10.0 + max(2.0, a*a)), 5000)
        vals = (grid**p) * np.exp(a*grid - b_eps*(grid**2))
        return float(vals.max())
    return sp.simplify(sp.Max(*val_candidates))

def kappa_n_symbolic_bound(n: int, eps_val: float) -> Tuple[sp.Expr, List[KappaPiece]]:
    """
    Produce a *symbolic* conservative bound κ_n(ε):
      Sum over nonlinear Faà di Bruno pieces of envelopes divided by w(ω)=ω^2 cosh(ω/2),
      evaluated on the smoothed spectrum (adds exp(-2ε ω^2)).
    Structure (conservative):
      Each product of r jets contributes roughly:
        ω^{n-2} * exp((r-1)|ω|/2) * exp(-2ε ω^2)
      with a combinatorial prefactor from the multiplicities.
    We return (kappa_expr, pieces) where pieces document the (p,a,b,C).
    NOTE: This is intentionally conservative and meant as a placeholder to audit.
    """
    assert n >= 2
    pieces: List[KappaPiece] = []
    # Pull Faà di Bruno multiplicities; ignore the single linear piece (r=1) which goes to K_lin.
    terms = faa_di_bruno_terms(n)
    # Empirical conservative envelope: for a term with r factors, we use a = (r-1)/2 and polynomial degree p = n-2
    # and overall b = 2 for the exp(-2ε ω^2) from |m̂_ε|^2.
    p = n - 2
    b = 2.0
    kappa_expr = 0
    for (mult, m_tuple, r) in terms:
        if r <= 1:
            continue  # linear handled separately
        a = 0.5 * (r - 1)
        # crude combinatorial constant (can be improved): absolute multiplier
        C = abs(mult)
        # Sup over ω of ω^p exp(a ω - 2ε ω^2)
        sup_expr = sup_poly_exp(p, a, b*eps_val)
        pieces.append(KappaPiece(p=p, a=float(a), b=b, C=float(C)))
        kappa_expr += C * sup_expr
    return sp.simplify(kappa_expr), pieces

# -----------------------------------------------------------------------------
# Utility: numeric evaluator wrappers
# -----------------------------------------------------------------------------

def eval_Pn_coeffs(n: int) -> List[float]:
    return [float(sp.N(c)) for c in Pn_coeffs(n)]

def eval_Pn_at(n: int, xvals: np.ndarray) -> np.ndarray:
    poly = sp.lambdify(x, Pn_poly(n), "numpy")
    return poly(xvals)

def eval_kappa_n(n: int, eps_val: float) -> float:
    kappa_expr, _ = kappa_n_symbolic_bound(n, eps_val)
    try:
        return float(sp.N(kappa_expr))
    except Exception:
        return float(sp.N(kappa_expr.subs({eps: eps_val})))
