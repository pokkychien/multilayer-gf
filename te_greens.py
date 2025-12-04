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


# ------------------------------------------------------------
# Level 1: q(k_parallel) per layer
# ------------------------------------------------------------
def compute_q_list(n_list, k0, kp):
    """
    Return q_list for all layers.
    q_j = sqrt(kp^2 - eps_j k0^2) / k0
    (TE/TM share q)
    """
    pass


# ------------------------------------------------------------
# Level 2: reflection recursion (TE)
# ------------------------------------------------------------
def compute_R_down_all(n_list, d_list, k0, kp):
    """
    Compute all downward reflection coefficients R_down[j].
    Meaning:
        Standing at the TOP of layer j, looking DOWNWARD
        into layers j, j+1, j+2, ...
    """
    pass


def compute_R_up_all(n_list, d_list, k0, kp):
    """
    Compute all upward reflection coefficients R_up[j].
    Meaning:
        Standing at the BOTTOM of layer j, looking UPWARD
        into layers j-1, j-2, ...
    """
    pass


# ------------------------------------------------------------
# Level 3: same-layer source amplitudes f1, f2 (local)
# ------------------------------------------------------------
def TE_f1f2_same_layer(q_list, R_down, R_up, layer_src, z_src, k0):
    """
    Compute f1 (downward) and f2 (upward) at z = z_src
    inside the *source* layer layer_src.

    Returns:
        f1_src, f2_src  (complex amplitudes)
    """
    pass


# ------------------------------------------------------------
# Level 4: propagate to another layer
# ------------------------------------------------------------
def propagate_down_TE(f1_src, q_list, R_down, R_up,
                      layer_src, layer_obs, k0, d_list):
    """
    Propagate downward amplitude f1 from layer_src → layer_obs.
    Returns f1 at the observation layer.

    Assumes layer_obs > layer_src.
    """
    pass


def propagate_up_TE(f2_src, q_list, R_down, R_up,
                    layer_src, layer_obs, k0, d_list):
    """
    Propagate upward amplitude f2 from layer_src → layer_obs.
    Returns f2 at the observation layer.

    Assumes layer_obs < layer_src.
    """
    pass


# ------------------------------------------------------------
# Level 5: assemble G_yy
# ------------------------------------------------------------
def gyy_TE(n_list, d_list,
           layer_src, z_src,
           layer_obs, z_obs,
           k0, kp):
    """
    Main TE Green's function:

        G_yy(layer_obs, layer_src; z_obs, z_src; k_parallel)

    High-level pseudocode:

        1. q_list = compute_q_list(...)
        2. R_down = compute_R_down_all(...)
        3. R_up   = compute_R_up_all(...)

        4. f1_src, f2_src = TE_f1f2_same_layer(...)

        5. if layer_obs == layer_src:
                # use same-layer closed form (local Gyy)
           elif layer_obs > layer_src:
                f1_at_obs = propagate_down_TE(...)
                # assemble Gyy using downward branch
           else:
                f2_at_obs = propagate_up_TE(...)
                # assemble Gyy using upward branch
    """
    pass
