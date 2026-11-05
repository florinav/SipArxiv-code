"""Two-strain cholera model in the plane: only B1, B2 diffuse. Five starts, profiles, and strain 2 invading the homogeneous resident of strain 1. Writes choFront.pdf next to this file.
Illustrative parameters. Needs numpy, matplotlib. Several minutes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mu, Lam = 0.1, 0.1
be, xi = (1.0, 1.2), (1.0, 1.0)
ga, de = (0.2, 0.2), (1.0, 1.0)
d = (1.0, 0.25)
S0 = Lam / mu
THETA = 1e-2

def speed(S, i):
    ls = np.geomspace(1e-3, 30, 4000)
    M = np.array([[-(mu + ga[i]), be[i] * S], [xi[i], -de[i]]])
    D = np.diag([0.0, d[i]])
    return min(np.linalg.eigvals(l ** 2 * D + M).real.max() / l for l in ls)

S1 = (mu + ga[0]) * de[0] / (be[0] * xi[0])
c1, c2, c21 = speed(S0, 0), speed(S0, 1), speed(S1, 1)
print("c1 %.3f c2 %.3f c21 %.3f" % (c1, c2, c21))

HALF, DX = 75.0, 0.5
g = np.arange(-HALF, HALF + DX / 2, DX)
X, Y = np.meshgrid(g, g, indexing="xy")

def lap(u):
    w = np.pad(u, 1, mode="edge")
    return (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2] - 4 * u) / DX ** 2

def disc(x0, y0, r):
    return np.hypot(X - x0, Y - y0) < r

L = ((X > -6) & (X < -2) & (Y > -6) & (Y < 6)) | ((X > -6) & (X < 6) & (Y > -6) & (Y < -2))
CASES = {
    "a": (L, L, (0, 0), (0, 0)),
    "b": (disc(-20, 0, 2.5), disc(0, 0, 15), (-20, 0), (0, 0)),
    "c": (disc(-25, -15, 4), disc(20, 15, 4), (-25, -15), (20, 15)),
    "d": (disc(0, 0, 14), disc(6, 4, 4), (0, 0), (6, 4)),
    "e": (disc(6, 4, 4), disc(0, 0, 14), (6, 4), (0, 0)),
}
marks = [10, 20, 30, 40, 50, 60]
TEND = 70.0

def run(p1, p2, c10, c20):
    S = np.full(X.shape, S0)
    I1, I2 = np.zeros(X.shape), np.zeros(X.shape)
    B1, B2 = np.where(p1, 0.1, 0.0), np.where(p2, 0.1, 0.0)
    dt = 0.2 * DX ** 2 / max(d)
    R1, R2 = np.hypot(X - c10[0], Y - c10[1]), np.hypot(X - c20[0], Y - c20[1])
    snap, rad, t, k = {}, [], 0.0, 0
    while t <= TEND + 1e-9:
        if k % 25 == 0:
            rad.append((t, R1[B1 > THETA].max() if (B1 > THETA).any() else 0.0, R2[B2 > THETA].max() if (B2 > THETA).any() else 0.0))
        for m in marks:
            if abs(t - m) < dt / 2:
                snap[m] = (B1.copy(), B2.copy(), S.copy())
        inf1, inf2 = be[0] * S * B1, be[1] * S * B2
        dS = Lam - mu * S - inf1 - inf2
        dI1, dI2 = inf1 - (mu + ga[0]) * I1, inf2 - (mu + ga[1]) * I2
        dB1 = d[0] * lap(B1) + xi[0] * I1 - de[0] * B1
        dB2 = d[1] * lap(B2) + xi[1] * I2 - de[1] * B2
        S, I1, I2, B1, B2 = S + dt * dS, I1 + dt * dI1, I2 + dt * dI2, B1 + dt * dB1, B2 + dt * dB2
        t += dt; k += 1
    return snap, np.array(rad)

res = {k: run(*v) for k, v in CASES.items()}
for k, (snap, rad) in res.items():
    for j, nm in ((1, "B1"), (2, "B2")):
        for a, b in ((5, 25), (25, 45), (45, 65)):
            w = (rad[:, 0] > a) & (rad[:, 0] < b) & (rad[:, j] < HALF - 3)
            if w.sum() > 3:
                print(k, nm, "t in (%d,%d) slope %.3f" % (a, b, np.polyfit(rad[w, 0], rad[w, j], 1)[0]), end="; ")
    print()

for k in "ab" + "e":
    rad = res[k][1]
    dif = rad[:, 1] - rad[:, 2]
    cr = [rad[i, 0] for i in range(1, len(dif)) if dif[i - 1] * dif[i] < 0 and rad[i, 0] > 5]
    print(k, "crossing times", [round(float(x), 1) for x in cr])
rad = res["a"][1]
for a, b in ((2, 8), (8, 14), (14, 20), (20, 26), (26, 32)):
    w = (rad[:, 0] > a) & (rad[:, 0] < b)
    print("a strain 2 slope on (%d,%d): %.3f" % (a, b, np.polyfit(rad[w, 0], rad[w, 2], 1)[0]))

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "font.family": "serif", "mathtext.fontset": "cm"})
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.7), constrained_layout=True)
TITLES = {"a": "(a) both from one patch", "b": "(b) strain 1 beside strain 2", "c": "(c) separate patches", "d": "(d) strain 2 inside strain 1", "e": "(e) strain 1 inside strain 2"}
for a_, k in zip([ax[0, 0], ax[0, 1], ax[0, 2], ax[0, 3], ax[1, 0]], "abcde"):
    snap, _ = res[k]
    p1, p2 = CASES[k][0], CASES[k][1]
    for m in marks:
        a_.contour(X, Y, snap[m][0], levels=[THETA], colors="tab:blue", linewidths=0.6)
        a_.contour(X, Y, snap[m][1], levels=[THETA], colors="tab:red", linewidths=0.6)
    a_.contour(X, Y, p1.astype(float), levels=[0.5], colors="black", linewidths=0.9)
    a_.contour(X, Y, p2.astype(float), levels=[0.5], colors="black", linewidths=0.9)
    a_.set_xlim(-70, 70); a_.set_ylim(-70, 70); a_.set_aspect("equal"); a_.set_xticks([-50, 0, 50]); a_.set_yticks([-50, 0, 50]); a_.set_title(TITLES[k])
ax[0, 0].plot([], [], color="black", label="start")
ax[0, 0].plot([], [], color="tab:blue", label="strain 1")
ax[0, 0].plot([], [], color="tab:red", label="strain 2")
ax[0, 0].legend(frameon=False, fontsize=5.5, loc="upper left", borderaxespad=0.1, handlelength=1.2)

def slopes(a_, rad, lines, title):
    a_.plot(rad[:, 0], rad[:, 1], color="tab:blue")
    a_.plot(rad[:, 0], rad[:, 2], color="tab:red")
    for val, j, t0, t1, tm, lab, dy in lines:
        tt = np.array([t0, t1])
        y0 = np.interp(tm, rad[:, 0], rad[:, j])
        a_.plot(tt, y0 + val * (tt - tm), ls="--", lw=0.8, color="black")
        a_.text(t1, y0 + val * (t1 - tm) + dy, lab, fontsize=7, va="bottom", ha="right")
    a_.set_xlabel("$t$"); a_.set_title(title); a_.set_ylim(0, 70); a_.set_box_aspect(1)
slopes(ax[1, 1], res["a"][1], [(c1, 1, 20, 68, 45, r"$c_1$", 2), (c2, 2, 2, 22, 10, r"$c_2$", -9), (c21, 2, 30, 68, 50, r"$c_{2|1}$", 2)], "(f) radii in (a)")
ax[1, 1].set_ylabel(r"$r_i(t)$")

# (g) what the two strains see: profiles of (a) along the positive zeta_1 axis
B1a, B2a, Sa = res["a"][0][60]
j0 = np.abs(g).argmin(); on = g >= 0
ax[1, 2].plot(g[on], Sa[j0, on], color="black", label="$S$")
ax[1, 2].plot(g[on], B1a[j0, on], color="tab:blue", label="$B_1$")
ax[1, 2].plot(g[on], B2a[j0, on], color="tab:red", label="$B_2$")
ax[1, 2].set_xlim(0, 70); ax[1, 2].set_ylim(0, 1.05); ax[1, 2].set_xlabel(r"$\zeta_1$"); ax[1, 2].set_box_aspect(1)
ax[1, 2].set_title(r"(g) (a) at $t=60$, $\zeta_2=0$"); ax[1, 2].legend(frameon=False, fontsize=5.5, loc="center right")

# (h) strain 2 into the homogeneous resident of strain 1, on a half line
def resident(dx=0.25, T=800.0, H=160.0, b0=1e-2):
    z = np.arange(0, H + dx / 2, dx); dt = 0.2 * dx ** 2 / max(d)
    I1s = (Lam - mu * S1) / (mu + ga[0]); B1s = xi[0] * I1s / de[0]
    S, I1, B1 = np.full_like(z, S1), np.full_like(z, I1s), np.full_like(z, B1s)
    I2, B2 = np.zeros_like(z), np.where(z < 4, b0, 0.0)
    def lap1(u):
        w = np.pad(u, 1, mode="edge")
        return (w[2:] + w[:-2] - 2 * u) / dx ** 2
    rec = []
    for k in range(int(round(T / dt)) + 1):
        if k % int(round(5 / dt)) == 0:
            rec.append((k * dt, z[B2 > THETA].max() if (B2 > THETA).any() else 0.0))
        f1, f2 = be[0] * S * B1, be[1] * S * B2
        dS = Lam - mu * S - f1 - f2
        dI1, dI2 = f1 - (mu + ga[0]) * I1, f2 - (mu + ga[1]) * I2
        dB1 = d[0] * lap1(B1) + xi[0] * I1 - de[0] * B1
        dB2 = d[1] * lap1(B2) + xi[1] * I2 - de[1] * B2
        S, I1, I2, B1, B2 = S + dt * dS, I1 + dt * dI1, I2 + dt * dI2, B1 + dt * dB1, B2 + dt * dB2
    return np.array(rec)
hh = resident()
w = (hh[:, 0] >= 400) & (hh[:, 0] <= 800)
print("resident invasion slope on (400,800): %.4f, linear %.4f" % (np.polyfit(hh[w, 0], hh[w, 1], 1)[0], c21))
ax[1, 3].plot(hh[:, 0], hh[:, 1], color="tab:red")
tt = np.array([100.0, 800.0]); y0 = np.interp(800.0, hh[:, 0], hh[:, 1])
ax[1, 3].plot(tt, y0 + c21 * (tt - 800.0), ls="--", lw=0.8, color="black")
ax[1, 3].text(790, y0 - 22, r"$c_{2|1}$", fontsize=7, ha="right")
ax[1, 3].set_xlabel("$t$"); ax[1, 3].set_xlim(0, 800); ax[1, 3].set_ylim(0, 100); ax[1, 3].set_box_aspect(1)
ax[1, 3].set_title("(h) strain 2 into resident 1")
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "choFront.pdf"), dpi=200)
print("saved")
