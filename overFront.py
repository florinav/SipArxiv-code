"""The three-population network with non-disjoint minimal siphons {1,2}, {2,3}, with equal diffusion d=1:
x1_t = Lap x1 - x1 + Phi_a(x1 + alpha x2), x2_t = Lap x2 - x2 + Phi_a(x2 + beta x1 x3), x3_t = Lap x3 - x3 + Phi_a(x3 + gamma x2).
Reference coefficients; couplings alpha, gamma of the table. The residents E23 (x1=s_a) and E12 (x3=s_a) occupy two territories,
and x2 starts at zero: a curved border in the plane to t=40, a straight border on a line to t=4000.
Writes overFront.pdf next to this file. Needs numpy, matplotlib, scipy. A few minutes.
"""
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = np.array([342, 368, 128, 1024]) / 343
al, ga, s = 2e-7, 5e-8, 0.5
b12 = (1 / a[0] - 1) ** 2 / (al * s)
be = 1.2 * b12

def Phi(q):
    q = np.maximum(q, 0)
    p = q * (a[0] + q * (a[1] + q * (a[2] + q * a[3])))
    return p / (1 + p)

# the small root of x2 = Phi(x2 + beta p) disappears at p_c = max over (0,1/8) of Phi^{-1}(x)-x
xs = np.linspace(1e-4, 0.124, 4000)
Pinv = np.array([brentq(lambda q: Phi(q) - x, 0, 5) for x in xs])
pc = (Pinv - xs).max()
print("ignition level beta x1 x3 = %.3e at x2=%.3f; with x1 x3 = 1/4: beta_c = %.3e; beta_12 = %.2f" % (pc, xs[(Pinv - xs).argmax()], 4 * pc, b12))

DX = 0.5
dt = 0.2 * DX ** 2

def reac(x1, x2, x3, b):
    return -x1 + Phi(x1 + al * x2), -x2 + Phi(x2 + b * x1 * x3), -x3 + Phi(x3 + ga * x2)

# ---- the plane, curved border
HALF = 60.0
g = np.arange(-HALF, HALF + DX / 2, DX)
X, Y = np.meshgrid(g, g, indexing="xy")
def lap(u):
    w = np.pad(u, 1, mode="edge")
    return (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2] - 4 * u) / DX ** 2

def run(b, T, marks):
    border = 8 * np.sin(Y / 12)
    x1, x3 = np.where(X > border, s, 0.0), np.where(X <= border, s, 0.0)
    x2 = np.zeros(X.shape)
    snap = {}
    for k in range(int(round(T / dt)) + 1):
        for m in marks:
            if k == int(round(m / dt)):
                snap[m] = (x1.copy(), x2.copy(), x3.copy())
        r1, r2, r3 = reac(x1, x2, x3, b)
        x1, x2, x3 = x1 + dt * (lap(x1) + r1), x2 + dt * (lap(x2) + r2), x3 + dt * (lap(x3) + r3)
    return snap

marks = [5, 10, 20, 30, 40]
snap = run(be, 40, marks)
small = run(1e-6, 40, [40])
print("table coupling beta=1e-6: max x2 at t=40 = %.2e" % small[40][1].max())

# ---- the line, straight border, long time
H1 = 800.0
z = np.arange(-H1, H1 + DX / 2, DX)
def lap1(u):
    w = np.pad(u, 1, mode="edge")
    return (w[2:] + w[:-2] - 2 * u) / DX ** 2

x1, x3 = np.where(z > 0, s, 0.0), np.where(z <= 0, s, 0.0)
x2 = np.zeros_like(z)
T1 = 4000.0
rec, prof = [], {}
for k in range(int(round(T1 / dt)) + 1):
    if k % int(round(10 / dt)) == 0:
        rec.append((k * dt, z[x2 > 0.25].max() if (x2 > 0.25).any() else np.nan, z[x3 > 0.25].max()))
    if k == int(round(20 / dt)):
        prof = (x1.copy(), x2.copy(), x3.copy())
    r1, r2, r3 = reac(x1, x2, x3, be)
    x1, x2, x3 = x1 + dt * (lap1(x1) + r1), x2 + dt * (lap1(x2) + r2), x3 + dt * (lap1(x3) + r3)
rec = np.array(rec)
w0 = (rec[:, 0] >= 40) & (rec[:, 0] <= 800)
kap = np.polyfit(np.sqrt(rec[w0, 0]), rec[w0, 1], 1)
print("x2 half-width fit on (40,800): %.3f sqrt(t) + %.2f" % tuple(kap))
for p, q in ((400, 800), (800, 1600), (1600, 2400), (2400, 3200), (3200, 4000)):
    w = (rec[:, 0] >= p) & (rec[:, 0] <= q)
    print("slopes on (%d,%d): x2 %.4f, x3 %.4f" % (p, q, np.polyfit(rec[w, 0], rec[w, 1], 1)[0], np.polyfit(rec[w, 0], rec[w, 2], 1)[0]))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.35), constrained_layout=True)
cols = plt.cm.plasma(np.linspace(0.05, 0.85, len(marks)))
for m, c in zip(marks, cols):
    ax[0].contour(X, Y, snap[m][1], levels=[0.25], colors=[c], linewidths=0.9)
ax[0].contour(X, Y, snap[40][0], levels=[0.25], colors="gray", linewidths=0.9, linestyles="dashed")
ax[0].set_xlim(-HALF, HALF); ax[0].set_ylim(-HALF, HALF); ax[0].set_aspect("equal")
ax[0].set_xlabel(r"$\zeta_1$"); ax[0].set_ylabel(r"$\zeta_2$"); ax[0].set_title(r"(a) $x_2=1/4$, $t=5,\ldots,40$")

on = np.abs(z) <= 60
x1p, x2p, x3p = prof
ax[1].semilogy(z[on], x1p[on], color="tab:blue", label="$x_1$")
ax[1].semilogy(z[on], x3p[on], color="tab:red", label="$x_3$")
ax[1].semilogy(z[on], x2p[on], color="tab:green", label="$x_2$")
ax[1].semilogy(z[on], be * x1p[on] * x3p[on], color="black", lw=0.8, label=r"$\beta x_1x_3$")
ax[1].axhline(pc, color="gray", ls="--", lw=0.8)
ax[1].set_ylim(1e-7, 60); ax[1].set_xlim(-60, 60); ax[1].set_xlabel("distance from the border")
ax[1].set_title("(b) profiles at $t=20$"); ax[1].legend(frameon=False, fontsize=6, loc="lower center", ncol=2)

ax[2].plot(rec[:, 0], rec[:, 1], color="tab:green", label="$x_2$")
ax[2].plot(rec[:, 0], rec[:, 2], color="tab:red", label="$x_3$")
ax[2].plot(rec[:, 0], kap[0] * np.sqrt(rec[:, 0]) + kap[1], color="black", ls="--", lw=0.8, label=r"fit to $\sqrt{t}$")
ax[2].set_xlim(0, T1); ax[2].set_ylim(0, 420); ax[2].set_xlabel("$t$"); ax[2].set_title(r"(c) half-width of $\{x_i>1/4\}$")
ax[2].legend(frameon=False, fontsize=6, loc="upper left")
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "overFront.pdf"), dpi=220)
print("saved")
