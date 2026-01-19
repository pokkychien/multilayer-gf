"""
k_parallel -> real-space (rho) integrator for multilayer Green's functions.

Design goal (your workflow):
    - Keep te_greens.py as the "engine" (you will keep editing it).
    - This file ONLY:
        (1) imports te_greens.gyy_TE (and later TM engines)
        (2) performs angular integration -> Bessel J0
        (3) performs k_parallel radial integral
        (4) plots results

Math (2D in-plane Fourier/Bessel transform):
    G_yy(rho) = (1/(2π)) ∫_0^{∞} k_parallel * J0(k_parallel*rho) * G_yy(k_parallel) dk_parallel

Notes:
    - In te_greens.py, q_list is dimensionless and exponentials use exp(q*k0*z).
      Here, kp has physical dimension (1/length), rho has length.
    - You choose k_parallel_max and quadrature settings.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# You will keep modifying te_greens.py; we import from it on purpose.
from te_greens import gyy_TE


def J0_series(x: np.ndarray) -> np.ndarray:
    """
    Minimal, dependency-free J0(x) approximation (series, stable for small/moderate x).

    J0(x) = Σ_{m=0}^∞ (-1)^m (x^2/4)^m / (m!)^2

    This is fine for plotting / debugging. If you later want faster & more accurate:
        - use scipy.special.j0
    """
    x = np.asarray(x, dtype=np.complex128)
    x2_over_4 = (x * x) / 4.0

    # Adaptive-ish truncation: more terms for larger |x|
    # (still cheap, but don't go crazy)
    max_abs = float(np.max(np.abs(x)))
    if max_abs < 5:
        M = 40
    elif max_abs < 20:
        M = 80
    else:
        M = 140

    out = np.zeros_like(x, dtype=np.complex128)
    term = np.ones_like(x, dtype=np.complex128)
    out += term
    for m in range(1, M):
        term *= (-x2_over_4) / (m * m)
        out += term
    return out


def gyy_TE_rho(
    n_list,
    d_list,
    layer_src: int,
    z_src: float,
    layer_obs: int,
    z_obs: float,
    k0: float,
    rho: float,
    k_parallel_max: float,
    num_k: int = 4000,
    use_trapz: bool = True,
) -> complex:
    """
    Compute real-space (rho) G_yy^TE by k_parallel integral (radial Bessel transform).

    Parameters
    ----------
    rho : float
        Cylindrical radius ρ.
    k_parallel_max : float
        Upper cutoff of k_parallel integral.
    num_k : int
        Number of k points on [0, k_parallel_max]. (Large helps oscillatory integrals.)
    use_trapz : bool
        If True: numpy.trapz. If False: simple Simpson (requires odd num_k).

    Returns
    -------
    complex
        G_yy(ρ) after angle integration.
    """
    kps = np.linspace(0.0, k_parallel_max, num_k, dtype=float)

    # Evaluate G(kp) from your engine
    Gkp = np.empty_like(kps, dtype=np.complex128)
    for i, kp in enumerate(kps):
        Gkp[i] = gyy_TE(
            n_list, d_list,
            layer_src, z_src,
            layer_obs, z_obs,
            k0, kp
        )

    # Angular integral -> J0
    J0 = J0_series(kps * rho)

    integrand = kps * J0 * Gkp
    pref = 1.0 / (2.0 * np.pi)

    if use_trapz:
        return pref * np.trapz(integrand, kps)

    # Simpson (odd N)
    if num_k % 2 == 0:
        raise ValueError("Simpson needs odd num_k. Set num_k to an odd integer or use_trapz=True.")
    h = (k_parallel_max - 0.0) / (num_k - 1)
    S = integrand[0] + integrand[-1] + 4.0 * np.sum(integrand[1:-1:2]) + 2.0 * np.sum(integrand[2:-2:2])
    return pref * (h / 3.0) * S


def demo_plot_TE():
    """
    Minimal demo plot: Re/Im of G_yy(ρ) for a fixed wavelength and layer positions.

    Edit the geometry freely; this script intentionally stays separate from te_greens.py.
    """
    # Geometry (edit)
    n_list = [1.0, 1.5, 1.3, 1.0]
    d_list = [0.0, 2000e-9, 1500e-9]

    # Source / obs (edit)
    layer_src = 1
    layer_obs = 1
    z_src = 200e-9
    z_obs = 800e-9

    # Wavelength -> k0
    wl = 650e-9
    k0 = 2.0 * np.pi / wl

    # k_parallel integration setup (edit)
    # Typical: some multiple of k0. For evanescent contributions, you may need larger.
    k_parallel_max = 5.0 * k0
    num_k = 6001  # odd recommended if you switch to Simpson

    rhos = np.linspace(0.0, 0.5e-6, 200)

    G_rho = np.array([
        gyy_TE_rho(
            n_list, d_list,
            layer_src, z_src,
            layer_obs, z_obs,
            k0,
            rho=float(r),
            k_parallel_max=k_parallel_max,
            num_k=num_k,
            use_trapz=True,
        )
        for r in rhos
    ], dtype=np.complex128)

    plt.figure()
    plt.plot(rhos * 1e6, np.real(G_rho), label="Re Gyy")
    plt.plot(rhos * 1e6, np.imag(G_rho), label="Im Gyy")
    plt.xlabel(r"$\rho$ (µm)")
    plt.ylabel(r"$G_{yy}^{TE}(\rho)$ (arb.)")
    plt.legend()
    plt.title("TE: Bessel-integrated $G_{yy}$ from te_greens.gyy_TE")
    plt.tight_layout()
    plt.show()
    # 再用同一份 G_rho 直接轉成 2D（不重算）
    plot_Grho_as_2D(G_rho, rhos, extent_um=0.5, N=401, which="real")

def plot_Grho_as_2D(G_rho, rhos, extent_um=0.5, N=401, which="real"):
    """
    Make a 2D map from radial data G(ρ) by coordinate transform + interpolation.

    extent_um: plot x,y in [-extent_um, +extent_um] (µm)
    N: grid points per axis
    which: "real" or "imag" or "abs"
    """
    # 1) build (x,y) grid in meters
    x = np.linspace(-extent_um, extent_um, N) * 1e-6
    y = np.linspace(-extent_um, extent_um, N) * 1e-6
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    # 2) choose which scalar to plot
    if which == "real":
        g1d = np.real(G_rho)
        label = "Re Gyy"
    elif which == "imag":
        g1d = np.imag(G_rho)
        label = "Im Gyy"
    elif which == "abs":
        g1d = np.abs(G_rho)
        label = "|Gyy|"
    else:
        raise ValueError('which must be "real", "imag", or "abs"')

    # 3) interpolate g(ρ) onto R
    # rhos is in meters already in your code
    # np.interp is fast (linear); outside range -> fill with 0
    G2 = np.interp(R.ravel(), rhos, g1d, left=0.0, right=0.0).reshape(R.shape)

    # 4) plot
    plt.figure()
    im = plt.imshow(
        G2,
        origin="lower",
        extent=[-extent_um, extent_um, -extent_um, extent_um],
        aspect="equal",
    )
    plt.colorbar(im, label=label)
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title(f"TE: {label} from radial Gyy(ρ)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_plot_TE()
