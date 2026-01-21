import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Bessel J0 (series, enough for plotting)
# ------------------------------------------------------------
def J0_series(x):
    x = np.asarray(x, dtype=np.complex128)
    x2_over_4 = (x * x) / 4.0

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


# ------------------------------------------------------------
# Free-space Gyy in k_parallel space
# ------------------------------------------------------------
def gyy_free_kp(kp, k0, dz):
    qk = np.sqrt(kp**2 - k0**2 + 0j)
    return np.exp(- qk * abs(dz)) / (2.0 * qk)


# ------------------------------------------------------------
# k_parallel → rho (Bessel integral)
# ------------------------------------------------------------
def Gyy_free_rho(rho, k0, dz, kp_max, num_k=4000):
    kps = np.linspace(0.0, kp_max, num_k)
    Gkp = gyy_free_kp(kps, k0, dz)
    J0  = J0_series(kps * rho)
    integrand = kps * J0 * Gkp
    return (1.0 / (2.0 * np.pi)) * np.trapz(integrand, kps)


# ------------------------------------------------------------
# 2D map from radial G(rho)
# ------------------------------------------------------------
def plot_Grho_as_2D(G_rho, rhos, extent_nm=1500, N=401):
    x = np.linspace(-extent_nm, extent_nm, N)
    y = np.linspace(-extent_nm, extent_nm, N)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    G2 = np.interp(R.ravel(), rhos, np.abs(G_rho), left=0.0, right=0.0)
    G2 = G2.reshape(R.shape)

    plt.figure(figsize=(6,5))
    im = plt.imshow(
        G2,
        origin="lower",
        extent=[-extent_nm, extent_nm, -extent_nm, extent_nm],
        aspect="equal",
        cmap="viridis"
    )
    plt.colorbar(im, label=r"$|G_{yy}^{\mathrm{free}}|$")
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    plt.title("Free space $|G_{yy}|$")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Demo
# ------------------------------------------------------------
def demo_free_space():
    # --- physical parameters (nm units) ---
    wl = 650.0          # wavelength (nm)
    k0 = 2.0 * np.pi / wl
    dz = 200.0          # |z - z'| (nm)

    kp_max = 5.0 * k0
    num_k  = 6001

    rhos = np.linspace(0.0, 500.0, 300)

    # --- compute radial Gyy ---
    G_rho = np.array([
        Gyy_free_rho(rho, k0, dz, kp_max, num_k)
        for rho in rhos
    ])

    # --- 1D plot ---
    plt.figure()
    plt.plot(rhos * 1e-3, np.real(G_rho), label="Re Gyy")
    plt.plot(rhos * 1e-3, np.imag(G_rho), label="Im Gyy")
    plt.xlabel(r"$\rho$ (µm)")
    plt.ylabel(r"$G_{yy}^{\mathrm{free}}$ (arb.)")
    plt.legend()
    plt.title("Free space $G_{yy}(\\rho)$")
    plt.tight_layout()
    plt.show()

    # --- 2D plot (ABS only, as you wanted) ---
    plot_Grho_as_2D(G_rho, rhos, extent_nm=1500, N=401)


if __name__ == "__main__":
    demo_free_space()