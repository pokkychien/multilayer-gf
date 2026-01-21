"""
TE Green's function engine for multilayer structures.

Goal:
    Compute G_yy(l_obs, l_src, z_obs, z_src, k_parallel)
    using a clean 5-level architecture:

        Level 1: q list (per layer)
        Level 2: TE reflection recursion (R_down, R_up)
        Level 3: same-layer f1, f2 amplitudes (local)
        Level 4: propagation between layers (down/up)
        Level 5: assemble full TE Green's function G_yy

This is a collaborative development skeleton.
"""


import numpy as np
from multilayer_reflectance import RF_multilayer, RB_multilayer  

# ------------------------------------------------------------
# Level 1: q(k_parallel) per layer
# ------------------------------------------------------------
def compute_q_list(n_list, k0, kp):
    """
    Return q_list for all layers.
    q_j = - sqrt(kp^2 - eps_j k0^2) / k0
    (TE/TM share q)

    n_list: list of refractive indices
    k0:     free-space wavenumber
    kp:     in-plane k_parallel
    """
    eps_list = np.array(n_list)**2

    # 使用陣列運算，一次算完所有 q_j
    q_list = np.sqrt(kp**2 - eps_list * k0**2 + 0j) / k0

    return q_list

def compute_interface_params_up_down(n_list, d_list, k0, kp, polarization="TE"):
    """
    Returns two lists:
        params_down[L] : for crossing interface L downward (layer L → L+1)
        params_up[L]   : for crossing interface L upward   (layer L+1 → L)

    Each params_down[L] / params_up[L] contains:
        q_up      : q in layer L      (source side)
        q_down    : q in layer L+1    (destination side)
        x         : exp(-q_up * k0 * d_list[L])  # SAME for up/down
        wp, wq    : jump coefficients for recursion
    """
    import numpy as np

    N = len(n_list)
    eps = np.array(n_list) ** 2
    kp2 = kp ** 2

    params_down = []
    params_up = []

    for L in range(N - 1):

        # --- q-values ---------------------------------------------------------
        q_up_layer   = np.sqrt(kp2 - eps[L]   * k0**2 + 0j) / k0
        q_down_layer = np.sqrt(kp2 - eps[L+1] * k0**2 + 0j) / k0

        # --- exponential factor (core fix) ------------------------------------
        # Always decay using q_up (layer where wave propagates before interface)
        xL = np.exp(- q_up_layer * k0 * d_list[L])

        # --- jump ratios -------------------------------------------------------
        if polarization.upper() == "TE":
            wn_down = q_down_layer / q_up_layer
            wn_up   = q_up_layer   / q_down_layer
        else:   # TM
            wn_down = (q_down_layer / q_up_layer) * (eps[L] / eps[L+1])
            wn_up   = (q_up_layer   / q_down_layer) * (eps[L+1] / eps[L])

        wm = 1.0

        wp_down = wm + wn_down
        wq_down = wm - wn_down

        wp_up = wm + wn_up
        wq_up = wm - wn_up

        # --- append results ----------------------------------------------------
        params_down.append({
            "q_up": q_up_layer,
            "q_down": q_down_layer,
            "x": xL,
            "wp": wp_down,
            "wq": wq_down
        })

        params_up.append({
            "q_up": q_up_layer,
            "q_down": q_down_layer,
            "x": xL,      # ★ SAME x for upward propagation
            "wp": wp_up,
            "wq": wq_up
        })

    return params_down, params_up
# ------------------------------------------------------------
# Level 2: reflection recursion (TE)
# ------------------------------------------------------------
def compute_RF_all(n_list, d_list, k0, kp, polarization="TE"):
    """
    RF_all[j] = reflection seen when standing at TOP of layer j, looking downward.
    Implemented using params_down (wp, wq, x), equivalent to original RF_multilayer().
    """
    params_down, params_up = compute_interface_params_up_down(
        n_list, d_list, k0, kp, polarization
    )
    N = len(n_list)

    RF_all = [0.0 + 0.0j] * N

    # bottom-most semi-infinite layer has no downward reflection
    r_eff = 0.0 + 0.0j

    # recursion from bottom to top
    # interfaces are L = 0...(N-2), between layer L and L+1
    for L in range(N-2, -1, -1):
        p = params_down[L]
        wp = p["wp"]
        wq = p["wq"]
        x  = p["x"]      # exponential through layer L (downward)

        # downward recursion formula:
        # r_new = x^2 * (wq + wp * r_old) / (wp + wq * r_old)
        num = wq + wp * r_eff
        den = wp + wq * r_eff
        r_eff = x * (num / den) * x   # x^2 * (bl/al)

        RF_all[L] = r_eff

    RF_all[N-1] = 0.0 + 0.0j   # lowest semi-infinite layer
    return RF_all


def compute_RB_all(n_list, d_list, k0, kp, polarization="TE"):
    """
    RB_all[j] = reflection seen when standing at BOTTOM of layer j, looking upward.
    Implemented using params_up (wp, wq, x), equivalent to original RB_multilayer().
    """
    params_down, params_up = compute_interface_params_up_down(
        n_list, d_list, k0, kp, polarization
    )
    N = len(n_list)

    RB_all = [0.0 + 0.0j] * N

    # top-most semi-infinite region has no upward reflection
    r_eff = 0.0 + 0.0j

    # recursion from top to bottom
    for L in range(0, N-1):
        p = params_up[L]
        wp = p["wp"]
        wq = p["wq"]
        x  = p["x"]      # exponential through layer L (upward)

        # upward recursion formula:
        # r_new = (wq + wp * x^2 * r_old) / (wp + wq * x^2 * r_old)
        xr = x * r_eff * x
        num = wq + wp * xr
        den = wp + wq * xr
        r_eff = num / den

        RB_all[L+1] = r_eff

    RB_all[0] = 0.0 + 0.0j   # top semi-infinite region
    return RB_all

# ------------------------------------------------------------
# Level 3: same-layer source amplitudes f1, f2 (local)
# ------------------------------------------------------------
def TE_f1yf2y_same_layer(q_list, R_down, R_up, layer_src, z_src, k0):
    """
    Compute TE source amplitudes (f1y, f2y) inside the source layer.

    f1y : downward-propagating TE wave amplitude
    f2y : upward-propagating TE wave amplitude
    """

    q  = q_list[layer_src]
    RF = R_down[layer_src]
    RB = R_up[layer_src]

    ez  = np.exp(+q * k0 * z_src)
    emz = np.exp(-q * k0 * z_src)

    den = 2 * q * k0 * (1 - RF * RB)

    f1y = ( ez  + emz * RB ) / den
    f2y = ( emz + ez  * RF ) / den

    return f1y, f2y


# ------------------------------------------------------------
# Level 4: propagate to another layer
# ------------------------------------------------------------
def propagate_down_TE(f1_src, q_list, R_down, R_up, layer_src, layer_obs, k0, d_list):
    if layer_obs <= layer_src:
        raise ValueError("propagate_down_TE requires layer_obs > layer_src")

    f = f1_src
    for l in range(layer_src, layer_obs):
        d_l = d_list[l] if l < len(d_list) else 0.0

        ql  = q_list[l]
        qlp = q_list[l+1]

        wn = qlp / ql
        wp = 1.0 + wn          # M_l^+
        wq = 1.0 - wn          # M_l^-

        Rnext = R_down[l+1]    # R_{l+1}

        T = 1.0 / (wp + wq * Rnext)       # T_l (paper)
        x = np.exp(-ql * k0 * d_l)        # e^{-q_l d_l}

        f = T * x * f                     # f_{l+1} = T_l e^{-q_l d_l} f_l

    return f


def propagate_up_TE(f2_src, q_list, R_up,
                    layer_src, layer_obs,
                    k0, d_list):
    """
    Paper-consistent TE upward propagation.

    Implements (paper notation):
        g_l = e^{-q_l d_l} ( m^+ + m^- e^{-q_l d_l} r_l e^{-q_l d_l} )^{-1} g_{l+1}

    Here:
        g ↔ f2
        r_l ↔ R_up[l]
        m^+ = 1 + q_l / q_{l+1}
        m^- = 1 - q_l / q_{l+1}

    Assumes:
        layer_obs < layer_src
    """
    if layer_obs >= layer_src:
        raise ValueError("propagate_up_TE requires layer_obs < layer_src")

    f = f2_src

    # l runs: layer_src-1, ..., layer_obs
    for l in range(layer_src - 1, layer_obs - 1, -1):

        d_l = d_list[l] if l < len(d_list) else 0.0

        ql   = q_list[l]
        qlp1 = q_list[l + 1]

        x = np.exp(-ql * k0 * d_l)

        wn = ql / qlp1
        mp = 1.0 + wn     # m^+
        mm = 1.0 - wn     # m^-

        r_l = R_up[l]

        T = 1.0 / (mp + mm * x * r_l * x)

        f = x * T * f

    return f


# ------------------------------------------------------------
# Level 5: assemble G_yy
# ------------------------------------------------------------
def gyy_TE(n_list, d_list,
           layer_src, z_src,
           layer_obs, z_obs,
           k0, kp):

    # -------------------------
    # Level 1: q per layer
    # -------------------------
    q_list = compute_q_list(n_list, k0, kp)

    # -------------------------
    # Level 2: reflections
    # -------------------------
    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TE")
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TE")

    # -------------------------
    # Level 3: source amplitudes
    # -------------------------
    f1y_src, f2y_src = TE_f1yf2y_same_layer(
        q_list, R_down, R_up, layer_src, z_src, k0
    )

    q_obs  = q_list[layer_obs]
    RF_obs = R_down[layer_obs]
    RB_obs = R_up[layer_obs]

    # ============================================================
    # Case A: same layer
    # ============================================================
    if layer_obs == layer_src:
        q  = q_list[layer_src]
        RF = R_down[layer_src]
        RB = R_up[layer_src]

        direct = (1.0 / (2.0 * q * k0)) * np.exp(
            -q * k0 * np.abs(z_obs - z_src)
        )

        cavity = (
            np.exp(+q * k0 * z_obs) * RF * f1y_src
            + np.exp(-q * k0 * z_obs) * RB * f2y_src
        )

        Gyy = direct + cavity

    # ============================================================
    # Case B: observation below source
    # ============================================================
    elif layer_obs > layer_src:
        f1y_at_obs_top = propagate_down_TE(
            f1y_src, q_list, R_down,
            layer_src, layer_obs, k0, d_list
        )

        Gyy = (
            np.exp(-q_obs * k0 * z_obs)
            + np.exp(+q_obs * k0 * z_obs) * RF_obs
        ) * f1y_at_obs_top

    # ============================================================
    # Case C: observation above source
    # ============================================================
    else:
        f2y_at_obs_top = propagate_up_TE(
            f2y_src, q_list, R_up,
            layer_src, layer_obs, k0, d_list
        )

        Gyy = (
            np.exp(+q_obs * k0 * z_obs)
            + np.exp(-q_obs * k0 * z_obs) * RB_obs
        ) * f2y_at_obs_top

    return Gyy


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_list = [1.0, 1.5, 1.3, 1.0]  # 四層結構
    d_list = [0.0, 2000, 1500]   # 四層結構

    wls = np.linspace(500, 900, 200)

    diff_RF = []
    diff_RB = []

    for pol in ["TE", "TM"]:
        diff_RF = []
        diff_RB = []

        for wl in wls:
            k0 = 2*np.pi / wl
            kp = 0.3*k0

            # OLD version
            RF_old = [RF_multilayer(n_list, d_list, pol, k0, kp, j)
                      for j in range(len(n_list))]
            RB_old = [RB_multilayer(n_list, d_list, pol, k0, kp, j)
                      for j in range(len(n_list))]

            # NEW version
            RF_new = compute_RF_all(n_list, d_list, k0, kp, pol)
            RB_new = compute_RB_all(n_list, d_list, k0, kp, pol)

            diff_RF.append(np.max(np.abs(np.array(RF_old) - np.array(RF_new))))
            diff_RB.append(np.max(np.abs(np.array(RB_old) - np.array(RB_new))))

        print(f"=== polarization: {pol} ===")
        print("max |RF_old - RF_new| =", np.max(diff_RF))
        print("max |RB_old - RB_new| =", np.max(diff_RB))
        print()
