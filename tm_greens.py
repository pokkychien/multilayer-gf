"""
TM Green's function engine for multilayer structures.

Goal:
    Compute TM Green's tensor components:
        g_xx, g_zz, g_xz, g_zx

This module REUSES the multilayer geometry and reflection machinery
defined in te_greens.py, to ensure identical definitions of:

    - q_list
    - interface parameters
    - RF / RB recursion

Only TM-specific source normalization, propagation, and assembly
are implemented here.
"""

import numpy as np

# ------------------------------------------------------------
# Import shared multilayer machinery (Level 1 & 2)
# ------------------------------------------------------------
from te_greens import (
    compute_q_list,
    compute_interface_params_up_down,
    compute_RF_all,
    compute_RB_all,
)

# ------------------------------------------------------------
# Level 3: same-layer TM source amplitudes
# ------------------------------------------------------------
def TM_f1xf2x_same_layer(q_list, R_down, R_up,
                         layer_src, z_src, k0, kp):
    """
    TM source amplitudes for g_xx component.

    Returns:
        f1x : downward-propagating TM amplitude
        f2x : upward-propagating TM amplitude
    """

    q  = q_list[layer_src]
    RF = R_down[layer_src]
    RB = R_up[layer_src]

    # TM needs epsilon of source layer
    # eps = n^2, but we don't recompute it here;
    # caller should ensure consistency
    # For same-layer source normalization:
    # e2 = eps[layer_src]
    # => infer via q definition: eps = (kp^2 - (q k0)^2) / k0^2
    # BUT: safer to pass eps_list in future
    # For now, match your old code exactly by assuming eps is known externally
    # → we reconstruct e2 from q and kp
    e2 = (kp**2 - (q * k0)**2) / (k0**2)

    ez  = np.exp(+q * k0 * z_src)
    emz = np.exp(-q * k0 * z_src)

    # denominator (exactly as in gxx_tm_in_film_multilayer)
    den = 2 * e2 * (k0 / q) * (1 - RF * RB)

    # TM g_xx amplitudes
    f1x = -(ez  + emz * RB) / den
    f2x = -(emz + ez  * RF) / den

    return f1x, f2x


def TM_f1zf2z_same_layer(q_list, R_down, R_up,
                         layer_src, z_src, k0, kp):
    """
    TM source amplitudes for g_zz component.

    Returns:
        f1z : downward-propagating TM amplitude
        f2z : upward-propagating TM amplitude
    """

    q  = q_list[layer_src]
    RF = R_down[layer_src]
    RB = R_up[layer_src]

    # reconstruct epsilon of source layer (same strategy as g_xx)
    e2 = (kp**2 - (q * k0)**2) / (k0**2)

    ez  = np.exp(+q * k0 * z_src)
    emz = np.exp(-q * k0 * z_src)

    # denominator exactly as in gzz_tm_in_film_multilayer
    den = (2 * k0**2 * e2 / (1j * kp)) * (1 - RF * RB)

    # TM g_zz source amplitudes
    f1z = ( ez  - emz * RB ) / den
    f2z = (-emz + ez  * RF ) / den

    return f1z, f2z


# ------------------------------------------------------------
# Level 4: TM propagation between layers
# ------------------------------------------------------------
def propagate_down_TM(f1_src, q_list, R_down,
                      layer_src, layer_obs,
                      k0, d_list, eps_list):
    """
    Propagate TM downward amplitude from layer_src -> layer_obs.

    Assumes:
        layer_obs > layer_src

    TM per-step propagation (paper-consistent):

        f_{l+1} = T_l * exp(-q_l k0 d_l) * f_l

    with
        wn_l = (q_{l+1}/q_l) * (eps_l/eps_{l+1})
        wp_l = 1 + wn_l     = M_l^+
        wq_l = 1 - wn_l     = M_l^-
        T_l  = 1 / (wp_l + wq_l * R_down[l+1])
    """

    if layer_obs <= layer_src:
        raise ValueError("propagate_down_TM requires layer_obs > layer_src")

    f = f1_src

    for l in range(layer_src, layer_obs):

        # thickness of layer l
        d_l = d_list[l] if l < len(d_list) else 0.0

        ql  = q_list[l]
        qlp = q_list[l + 1]

        eps_l  = eps_list[l]
        eps_lp = eps_list[l + 1]

        # TM interface coefficient
        wn = (qlp / ql) * (eps_l / eps_lp)

        wp = 1.0 + wn        # M_l^+
        wq = 1.0 - wn        # M_l^-

        # downward-looking reflection from layer l+1
        Rnext = R_down[l + 1]

        # transmission coefficient (paper form, NO extra factor)
        T = 1.0 / (wp + wq * Rnext)

        # propagation through layer l
        x = np.exp(-ql * k0 * d_l)

        # update amplitude
        f = T * x * f

    return f


def propagate_up_TM(f2_src, q_list, R_up,
                    layer_src, layer_obs,
                    k0, d_list, eps_list):
    """
    Paper-consistent TM upward propagation.

    Implements:
        f_l = e^{-q_l d_l} ( m^+ + m^- e^{-q_l d_l} R_l e^{-q_l d_l} )^{-1} f_{l+1}

    Assumes:
        layer_obs < layer_src
    """
    if layer_obs >= layer_src:
        raise ValueError("propagate_up_TM requires layer_obs < layer_src")

    f = f2_src

    # l = layer_src-1, ..., layer_obs
    for l in range(layer_src - 1, layer_obs - 1, -1):

        d_l = d_list[l] if l < len(d_list) else 0.0

        ql   = q_list[l]
        qlp1 = q_list[l + 1]

        eps_l  = eps_list[l]
        eps_lp = eps_list[l + 1]

        # TM interface coefficient
        wn = (ql / qlp1) * (eps_lp / eps_l)

        mp = 1.0 + wn     # m^+
        mm = 1.0 - wn     # m^-

        r_l = R_up[l]

        x = np.exp(-ql * k0 * d_l)

        T = 1.0 / (mp + mm * x * r_l * x)

        f = x * T * f

    return f


# ------------------------------------------------------------
# Level 5: assemble TM Green's function components
# ------------------------------------------------------------
def gxx_TM_same_layer(q_list, R_down, R_up,
                      layer, z_obs, z_src,
                      f1x_src, f2x_src,
                      k0, eps_list):
    """
    TM same-layer G_xx, strictly following paper.

    NOTE:
        This is SAME LAYER ONLY.
        Cross-layer formula is NOT written here.
    """

    q   = q_list[layer]
    eps = eps_list[layer]

    RF = R_down[layer]
    RB = R_up[layer]

    # --- paper first term ---
    direct = (
        - q / (2.0 * k0**2 * eps)
        * np.exp(-q * k0 * np.abs(z_obs - z_src))
    )

    # --- reflected terms ---
    refl = (
        np.exp(+q * k0 * z_obs) * RF * f1x_src
        + np.exp(-q * k0 * z_obs) * RB * f2x_src
    )

    return direct + refl


def gzz_TM(n_list, d_list,
           layer_src, z_src,
           layer_obs, z_obs,
           k0, kp):
    """
    TM Green's function component G_zz.

    Same-layer only.

    TODO:
        Cross-layer propagation form not written here.
        Ask 張亞中 how to extend G_zz to layer_obs != layer_src.
    """

    # Level 1
    q_list = compute_q_list(n_list, k0, kp)
    eps_list = np.array(n_list)**2

    # Level 2
    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TM")
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TM")

    if layer_obs != layer_src:
        raise NotImplementedError("gzz_TM: cross-layer case needs 張亞中確認")

    n = layer_src
    q = q_list[n]
    eps = eps_list[n]

    RF = R_down[n]
    RB = R_up[n]

    # Level 3: TM same-layer source amplitudes (paper form)
    f1z, f2z = TM_f1zf2z_same_layer(
        q_list, R_down, R_up,
        layer_src, z_src, k0, kp
    )

    # same-layer paper structure
    Gzz = (
        (kp**2) / (2 * k0**3 * eps * q)
        - (1j * kp / (q * k0)) * np.exp(+q * k0 * z_obs) * RF * f1z
        + (1j * kp / (q * k0)) * np.exp(-q * k0 * z_obs) * RB * f2z
    )

    return Gzz


def gxz_TM(n_list, d_list,
           layer_src, z_src,
           layer_obs, z_obs,
           k0, kp):
    """
    TM Green's function component G_xz.

    Same-layer only (layer_obs == layer_src).

    TODO (ask 張亞中):
        Paper does not give an explicit cross-layer (layer_obs != layer_src) formula for G_xz.
        Confirm how to construct/propagate the needed amplitudes for different layers.
    """

    # -------------------------
    # Level 1: q per layer
    # -------------------------
    q_list = compute_q_list(n_list, k0, kp)

    # -------------------------
    # Level 2: reflections (TM)
    # -------------------------
    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TM")  # RF_all
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TM")  # RB_all

    # -------------------------
    # Same-layer only
    # -------------------------
    if layer_obs != layer_src:
        raise NotImplementedError("gxz_TM: cross-layer case needs 張亞中確認")

    n = layer_src
    q  = q_list[n]
    RF = R_down[n]
    RB = R_up[n]

    sgn = np.sign(z_obs - z_src)

    # paper same-layer structure (written in your q-convention)
    Gxz = (-1j * kp / (2.0 * k0**2)) * (
        sgn * np.exp(-q * k0 * np.abs(z_obs - z_src))
        + np.exp(+q * k0 * z_obs) * RB * np.exp(-q * k0 * z_src)
        + np.exp(-q * k0 * z_obs) * RF * np.exp(+q * k0 * z_src)
    )

    return Gxz


def gzx_TM(n_list, d_list,
           layer_src, z_src,
           layer_obs, z_obs,
           k0, kp):
    """
    TM Green's function component g_zx.

    NOTE:
        - Only SAME-LAYER (layer_obs == layer_src) is implemented
        - Cross-layer formula not given explicitly in Y.C. Chang paper
        - NEED TO ASK 張亞中
    """

    if layer_obs != layer_src:
        raise NotImplementedError(
            "gzx_TM for different layers not implemented; ask Y.C. Chang"
        )

    eps_list = np.array(n_list)**2
    q_list = -np.sqrt(kp**2 - eps_list * k0**2 + 0j) / k0

    q  = q_list[layer_src]
    RF = compute_RF_all(n_list, d_list, k0, kp, polarization="TM")[layer_src]
    RB = compute_RB_all(n_list, d_list, k0, kp, polarization="TM")[layer_src]

    # TM source amplitudes (same layer)
    f1, f2 = TM_f1xf2x_same_layer(
        q_list, 
        compute_RF_all(n_list, d_list, k0, kp, "TM"),
        compute_RB_all(n_list, d_list, k0, kp, "TM"),
        layer_src, z_src, k0, kp
    )

    # exponentials
    ez   = np.exp(+q * k0 * z_obs)
    emz  = np.exp(-q * k0 * z_obs)
    ezp  = np.exp(+q * k0 * z_src)
    emzp = np.exp(-q * k0 * z_src)

    # sign term
    sgn = np.sign(z_obs - z_src)

    # === paper Eq.(17), NO rearrangement ===
    gzx = (
        -1j * kp / (2 * k0**2) * sgn * np.exp(-q * k0 * np.abs(z_obs - z_src))
        + (1j * kp / q) * (
            - ez * RF * f1
            + emz * RB * f2
        )
    )

    return gzx