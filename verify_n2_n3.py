import os
import numpy as np
import matplotlib.pyplot as plt

from src.li_framework import (
    gaussian_smoother, energy_E2, lambda_n_linear, positivity_margin
)
from src.kernels import weight_w, K_lin_symbol

# ----- test functions m(t) and their Fourier transforms mhat(omega)

def mhat_gaussian(omega, sigma=1.0):
    """
    Fourier transform of a real-space Gaussian m(t) ~ exp(-t^2/(2*sigma_t^2))
    If we take mhat(omega) directly as a Gaussian envelope:
      mhat(omega) = exp(-omega^2 / (2*sigma^2))
    that's fine for testing quadratic forms.
    """
    return np.exp(-(omega**2)/(2.0*sigma**2))

def mhat_lorentzian(omega, gamma=1.0):
    return 1.0 / (1.0 + (omega/gamma)**2)

def mhat_bump(omega):
    # compact-ish smooth bump (not truly compact support, but fast decay)
    return np.exp(-omega**2) * (1.0 + 0.1*np.cos(3*omega))

def ensure_out():
    if not os.path.exists("out"):
        os.makedirs("out")

def run_one(mhat_func, label, eps=0.5, omega_max=10.0, N=20001):
    omega = np.linspace(-omega_max, omega_max, N)
    mhat = mhat_func(omega)

    # energies and linear parts
    E = energy_E2(mhat, omega)
    L2 = lambda_n_linear(2, mhat, omega)
    L3 = lambda_n_linear(3, mhat, omega)

    # positivity margins with explicit kappa2, kappa3
    out2 = positivity_margin(2, mhat, omega, eps=eps, delta=0.2)
    out3 = positivity_margin(3, mhat, omega, eps=eps, delta=0.2)

    print(f"\n=== {label} ===")
    print(f"E2[m]        = {E:.6e}")
    print(f"L2[m]        = {L2:.6e}")
    print(f"L3[m]        = {L3:.6e}")
    print(f"lambda2 >=    {out2['lower_bound']:.6e}  (kappa2={out2['kappa']:.3f}, E2[m_eps]={out2['E_eps']:.6e})")
    print(f"lambda3 >=    {out3['lower_bound']:.6e}  (kappa3={out3['kappa']:.3f}, E2[m_eps]={out3['E_eps']:.6e})")

    # plots: weight, kernel symbols, and |mhat|^2 overlay
    ensure_out()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(omega, weight_w(omega), lw=1, label=r"$w(\omega)$")
    ax.set_yscale("log")
    ax.set_title("Weight $w(\\omega)$")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("value (log)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"out/weight_{label}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    K2 = K_lin_symbol(2, omega)
    K3 = K_lin_symbol(3, omega)
    ax.plot(omega, K2, lw=1, label=r"$\widehat K^{lin}_2$")
    ax.plot(omega, K3, lw=1, label=r"$\widehat K^{lin}_3$")
    ax.set_yscale("log")
    ax.set_title("Linear kernels")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("value (log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"out/kernels_{label}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(omega, np.abs(mhat)**2, lw=1, label=r"$| \widehat m |^2$")
    ax.plot(omega, np.abs(gaussian_smoother(omega, eps)*mhat)**2, lw=1, label=rf"$| \widehat m_\varepsilon |^2,\, \varepsilon={eps}$")
    ax.set_yscale("log")
    ax.set_title(f"Spectrum of test function ({label})")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("power (log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"out/spectra_{label}.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    # Choose a moderate smoothing (used only in the bound terms), you can tune this
    eps = 0.5

    run_one(lambda w: mhat_gaussian(w, sigma=1.0), label="gauss_s1", eps=eps)
    run_one(lambda w: mhat_gaussian(w, sigma=0.5), label="gauss_s0p5", eps=eps)
    run_one(lambda w: mhat_lorentzian(w, gamma=1.0), label="lorentz_g1", eps=eps)
    run_one(mhat_bump, label="bump", eps=eps)

    print("\nPlots saved to ./out/. You can adjust eps, omega_max, and test functions as needed.")
