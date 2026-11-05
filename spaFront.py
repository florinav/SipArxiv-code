"""Fronts in the plane for x_t = D Lap x - V x + Phi(W x), Phi(q) = L q / (1 + L q), L = 2 (data of Remark r:SpdAtt).
Writes spaFront.pdf next to this file. Needs numpy, matplotlib. About a minute.
The drift panel uses Q = q I, for which z(t, zeta) = z0(t, zeta + q t nu) exactly.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = np.array([1.0, 0.1])
V = np.array([[1.0, 0.0], [-0.5, 0.7]])
W = np.array([[0.2, 0.9], [0.3, 0.1]])
L = 2.0
HALF, DX = 26.0, 0.2
T_MARK = [0, 2, 4, 6, 8, 10]
LEVEL = 0.1

def phi(q):
    q = np.maximum(q, 0)
    return L * q / (1 + L * q)

def cbar():
    ls = np.geomspace(1e-3, 30, 3000)
    return min(np.linalg.eigvals(l ** 2 * np.diag(D) + L * W - V).real.max() / l for l in ls)

def lap(u):
    w = np.pad(u, 1, mode="edge")
    return (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2] - 4 * u) / DX ** 2

g = np.arange(-HALF, HALF + DX / 2, DX)
X, Y = np.meshgrid(g, g, indexing="xy")
# an irregular start: an L-shaped patch and a separate small square
patch = (((X > -9) & (X < -6) & (Y > -8) & (Y < -1)) | ((X > -9) & (X < -2) & (Y > -8) & (Y < -5)) | ((X > 3) & (X < 5) & (Y > 3) & (Y < 5)))
u = np.zeros((2,) + X.shape)
u[:, patch] = 1.0
dt = 0.2 * DX ** 2 / D.max()
snap = {}
t, k = 0.0, 0
while t <= T_MARK[-1] + 1e-9:
    for m in T_MARK:
        if abs(t - m) < dt / 2:
            snap[m] = u[0].copy()
    q = W[0, 0] * u[0] + W[0, 1] * u[1], W[1, 0] * u[0] + W[1, 1] * u[1]
    r0 = -V[0, 0] * u[0] - V[0, 1] * u[1] + phi(q[0])
    r1 = -V[1, 0] * u[0] - V[1, 1] * u[1] + phi(q[1])
    u = np.array([u[0] + dt * (D[0] * lap(u[0]) + r0), u[1] + dt * (D[1] * lap(u[1]) + r1)])
    t += dt

c = cbar()
print("cbar", c, "top", snap[T_MARK[-1]].max())
# radial reach of the level set, from the centre of the region at the last two times
for a, b in ((6, 10), (8, 10)):
    rad = []
    for m in (a, b):
        idx = np.where(snap[m] > LEVEL)
        rad.append((X[idx].max(), -X[idx].min(), Y[idx].max(), -Y[idx].min()))
    print("reach speed %d-%d, +x,-x,+y,-y:" % (a, b), [round((rad[1][i] - rad[0][i]) / (b - a), 3) for i in range(4)])

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.3), constrained_layout=True)
lim = 24
im = ax[0].pcolormesh(X, Y, snap[6], shading="auto", cmap="viridis", rasterized=True, vmin=0)
ax[0].set_title(r"$x_1$ at $t=6$")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)
colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(T_MARK)))
for m, col in zip(T_MARK, colors):
    ax[1].contour(X, Y, snap[m], levels=[LEVEL], colors=[col], linewidths=0.9)
ax[1].set_title("level curves, $t=0,2,\\ldots,10$")
q0 = 0.75
for m, col in zip(T_MARK, colors):
    ax[2].contour(X - q0 * m, Y, snap[m], levels=[LEVEL], colors=[col], linewidths=0.9)
ax[2].set_title(r"same, with drift $q=0.75$")
for a in ax:
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
    a.set_xlabel(r"$\zeta_1$"); a.set_xticks([-20, 0, 20]); a.set_yticks([-20, 0, 20])
ax[0].set_ylabel(r"$\zeta_2$")
ax[2].set_xlim(-32, 16); ax[2].set_xticks([-30, -15, 0, 15])
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spaFront.pdf"), dpi=200)
print("saved")
