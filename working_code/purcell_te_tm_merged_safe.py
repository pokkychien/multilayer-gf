#!/usr/bin/env python3
# Safe merged Purcell factor code: TE (gyy) + TM (gxx + gzz)
# TE block is kept IDENTICAL to the user's reference implementation.

import numpy as np
import matplotlib.pyplot as plt
from multilayer_reflectance import RF_multilayer, RB_multilayer

# ---------- TE Fresnel + gyy (in-film, TE only) ----------
def r_te(qi, qj):
    """TE Fresnel reflection (using q = sqrt(kp**2 - eps*k0**2)/k0)."""
    return (qi - qj) / (qi + qj)


def gyy_te_in_film_multilayer(n_list, d_list, target_layer, k0, kp, z):
    """
    Green's function component gyy (TE) at position z inside the target layer of a multilayer stack.
    n_list: list of refractive indices [n0, n1, ..., nN] (N+1 layers)
    d_list: list of thicknesses [d1, ..., dN] (N layers, does not include semi-infinite claddings)
    target_layer: index of the layer where z is located (0-based, matches n_list/d_list convention)
    k0: free-space wavenumber
    kp: in-plane wavevector
    z: position inside the target layer (0 <= z <= d_list[target_layer])
    """
    eps_list = np.array(n_list)**2
    q_list = -np.sqrt(kp**2 - eps_list*k0**2 + 0j) / k0
    RB = RB_multilayer(n_list, d_list, "TE", k0, kp, target_layer)
    RF = RF_multilayer(n_list, d_list,"TE", k0, kp, target_layer)

    q2 = q_list[target_layer]

    den = 2*q2*k0 * (1 - RF*RB)
    ez  = np.exp(+q2*k0*z)
    emz = np.exp(-q2*k0*z)

    f1 = ( ez  + emz*RB ) / den
    f2 = ( emz + ez*RF ) / den

    gyy = (1/(2*q2*k0)) + emz*RB*f2 + ez*RF*f1
    return gyy

# ---------- Purcell factor: Gauss-Legendre (TE, multilayer) ----------
def purcell_TE_multilayer(n_list, d_list, target_layer, wl, z, kp_max_factor=3.0, n_gauss=200):
    k0 = 2*np.pi / wl
    a, b = 0.0, kp_max_factor*k0
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)  # nodes on [-1,1]
    kp = 0.5*(xg + 1.0)*(b - a) + a
    w  = 0.5*(b - a)*wg
    gvals = np.array([gyy_te_in_film_multilayer(n_list, d_list, target_layer, k0, kpi, z) for kpi in kp])
    integrand = kp * np.imag(gvals)
    PF = (1/(2*np.pi)) * np.sum(w * integrand) / k0
    return PF


# ---------------- TM Fresnel + Green’s functions ----------------
def r_tm(qi, qj, ei, ej):
    """TM Fresnel reflection (using q = sqrt(kp**2 - eps*k0**2)/k0)."""
    return (qi/ei - qj/ej) / (qi/ei + qj/ej)


def gxx_tm_in_film_multilayer(n_list, d_list, target_layer, k0, kp, z):
    """
    Green's function component gxx (TM) at position z inside the target layer of a multilayer stack.
    n_list: list of refractive indices [n0, n1, ..., nN] (N+1 layers)
    d_list: list of thicknesses [d1, ..., dN] (N layers, does not include semi-infinite claddings)
    target_layer: index of the layer where z is located (0-based, matches n_list/d_list convention)
    k0: free-space wavenumber
    kp: in-plane wavevector
    z: position inside the target layer (0 <= z <= d_list[target_layer])
    """
    eps_list = np.array(n_list)**2
    q_list = -np.sqrt(kp**2 - eps_list*k0**2 + 0j) / k0

    # Compute reflection coefficients for multilayer stack (TM polarization)
    RB = RB_multilayer(n_list, d_list, "TM", k0, kp, target_layer)
    RF = RF_multilayer(n_list, d_list, "TM", k0, kp, target_layer)

    e2 = eps_list[target_layer]
    q2 = q_list[target_layer]

    den = 2 * e2 * (k0 / q2) * (1 - RF * RB)
    ez  = np.exp(+q2 * k0 * z)
    emz = np.exp(-q2 * k0 * z)

    f1 = -(ez  + emz * RB) / den
    f2 = -(emz + ez  * RF) / den

    gxx = -(q2 / (2 * k0 * e2)) + emz * RB * f2 + ez * RF * f1
    return gxx

def gzz_tm_in_film_multilayer(n_list, d_list, target_layer, k0, kp, z):
    """
    Green's function component gzz (TM) at position z inside the target layer of a multilayer stack.
    n_list: list of refractive indices [n0, n1, ..., nN] (N+1 layers)
    d_list: list of thicknesses [d1, ..., dN] (N layers, does not include semi-infinite claddings)
    target_layer: index of the layer where z is located (0-based, matches n_list/d_list convention)
    k0: free-space wavenumber
    kp: in-plane wavevector
    z: position inside the target layer (0 <= z <= d_list[target_layer])
    """
    eps_list = np.array(n_list)**2
    q_list = -np.sqrt(kp**2 - eps_list*k0**2 + 0j) / k0

    # Compute reflection coefficients for multilayer stack (TM polarization)
    RB = RB_multilayer(n_list, d_list, "TM", k0, kp, target_layer)
    RF = RF_multilayer(n_list, d_list, "TM", k0, kp, target_layer)

    e2 = eps_list[target_layer]
    q2 = q_list[target_layer]

    den = (2 * k0**2 * e2 / (1j * kp)) * (1 - RF * RB)
    ez  = np.exp(+q2 * k0 * z)
    emz = np.exp(-q2 * k0 * z)

    f1 = (ez  - emz * RB) / den
    f2 = (-emz + ez  * RF) / den

    gzz = (kp**2 / (2 * k0**3 * e2 * q2)) \
          - (1j * kp / (q2*k0)) * (ez * RF * f1) \
          + (1j * kp / (q2*k0)) * (emz * RB * f2)
    return gzz


def purcell_TM_multilayer(n_list, d_list, target_layer, wl, z, kp_max_factor=5.0, n_gauss=200):
    k0 = 2 * np.pi / wl
    a, b = 0.0, kp_max_factor * k0
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    kp = 0.5 * (xg + 1.0) * (b - a) + a
    w = 0.5 * (b - a) * wg
    gvals = np.array([
        gxx_tm_in_film_multilayer(n_list, d_list, target_layer, k0, kpi, z) +
        gzz_tm_in_film_multilayer(n_list, d_list, target_layer, k0, kpi, z)
        for kpi in kp
    ])
    integrand = kp * np.imag(gvals)
    PF = (1 / (2 * np.pi)) * np.sum(w * integrand) / k0
    return PF

# ---------------- main ----------------
def main():
    # ----- parameters -----
    n1, n2, n3 = 1.0, 1.7, 1.3
    d  = 1000e-9
    z  = 0.9 * d
    wl_min, wl_max, points = 500e-9, 1000e-9, 300
    kp_max_factor = 5.0

    # scan wavelength
    wls = np.linspace(wl_min, wl_max, points)
    PF_te = np.empty(points)
    PF_tm = np.empty(points)

    for i, wl in enumerate(wls):
        PF_te[i] = purcell_TE_multilayer([n1, n2, n3, n2, n1], [0, d/2, d, d/2], 2, wl, z,
                         kp_max_factor=kp_max_factor, n_gauss=300)
        PF_tm[i] = purcell_TM_multilayer([n1, n2, n3, n2, n1], [0, d/2, d, d/2], 2, wl, z,
                         kp_max_factor=kp_max_factor, n_gauss=300)


    # plot TE vs lambda
    plt.figure(figsize=(9,5))
    plt.plot(wls*1e9, PF_te, label="PF_TE (gyy)", color="blue")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Purcell factor (TE)")
    plt.title(f"Purcell factor TE @ z={z/d:.2f} d, d={d*1e9:.0f} nm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("PF_TE_vs_lambda.png", dpi=150)
    plt.show()

    # plot TM vs lambda
    plt.figure(figsize=(9,5))
    plt.plot(wls*1e9, PF_tm, label="PF_TM (gxx+gzz)", color="red")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Purcell factor (TM)")
    plt.title(f"Purcell factor TM @ z={z/d:.2f} d, d={d*1e9:.0f} nm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("PF_TM_vs_lambda.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
