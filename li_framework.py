import numpy as np
from .kernels import weight_w, K_lin_symbol

def gaussian_smoother(omega: np.ndarray, eps: float) -> np.ndarray:
    """
    Fourier multiplier for Gaussian smoothing of variance eps.
    """
    return np.exp(-eps * omega**2)

def energy_E2(mhat: np.ndarray, omega: np.ndarray) -> float:
    """
    E2[m] = (1/2π) ∫ w(ω) |m̂(ω)|^2 dω
    """
    w = weight_w(omega)
    integrand = w * (np.abs(mhat)**2)
    return (1.0/(2*np.pi)) * np.trapz(integrand, omega)

def lambda_n_linear(n: int, mhat: np.ndarray, omega: np.ndarray) -> float:
    """
    Linear part L_n[m] = (1/2π) ∫ K_lin_n(ω) |m̂(ω)|^2 dω
    """
    Klin = K_lin_symbol(n, omega)
    integrand = Klin * (np.abs(mhat)**2)
    return (1.0/(2*np.pi)) * np.trapz(integrand, omega)

# --- Reference bounds for kappa_n(eps) we used explicitly for n=2,3
def kappa2(eps: float) -> float:
    # |R_2[m_eps]| ≤ 2 * exp(1/(32 eps)) * E2[m_eps]
    return 2.0 * np.exp(1.0/(32.0*eps))

def kappa3(eps: float) -> float:
    # ≤ (3/(8 eps)) exp(1/(32 eps)) + (1/(16 eps)) exp(1/(8 eps))
    return (3.0/(8.0*eps))*np.exp(1.0/(32.0*eps)) + (1.0/(16.0*eps))*np.exp(1.0/(8.0*eps))

def positivity_margin(n: int, mhat: np.ndarray, omega: np.ndarray, eps: float, delta: float=0.2):
    """
    Compute:
      L_n[m]  (linear positive part)
      E2[m]   (energy)
      E2[m_eps] with Gaussian smoothing eps and contraction check
      lower bound on lambda_n ≈ L_n[m] - kappa_n(eps) * E2[m_eps]
    For n=2,3 we use the explicit kappa_n formulas. For other n, returns None.
    """
    L = lambda_n_linear(n, mhat, omega)
    E = energy_E2(mhat, omega)

    g = gaussian_smoother(omega, eps)
    mhat_eps = g * mhat
    E_eps = energy_E2(mhat_eps, omega)

    if n == 2:
        kappa = kappa2(eps)
    elif n == 3:
        kappa = kappa3(eps)
    else:
        return {"n": n, "L": L, "E": E, "E_eps": E_eps, "kappa": None, "lower_bound": None}

    lower = L - kappa * E_eps
    return {"n": n, "L": L, "E": E, "E_eps": E_eps, "kappa": kappa, "lower_bound": lower}
