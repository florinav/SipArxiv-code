"""Numbers of Remark r:FroSta, the stability of the fronts. Needs numpy, scipy. A few minutes; prints only.
Scalar quartic x_t = x_zz - x + Phi_a(x), reference coefficients: speed by shooting in the phase plane (u, E = u'^2/2), from
u = 1/2 - 1e-3 on the unstable direction to u = 2e-3 on the stable one, bisection on c. Eigenvalues of the self-adjoint form
v'' + (f'(U) - c^2/4) v = mu v on z in [-200,700], second differences with step 0.1, v = 0 at both ends. Evans function:
Wronskian at z = 0 of p_+-, with the factor exp(r z) removed, RK4 with step 0.1 from z = 700 and z = -200.
Systems, at prescribed c > c_*: profile on z in [-100,40], step 0.05, by implicit pseudo-time stepping; central differences for
diffusing components, upwind differences for nondiffusing ones; no flux at z = -100, end value at z = 40 fixed from the
exponential traveling mode with the smaller root lam of c(lam) = c. Perturbations vanish beyond z = 40.
Eigenvalues after the similarity exp(lam_* max(z,0)) ("weighted"); "plain" shows the round-off without it.
Optional arguments: step and right end of the interval, for the refinement checks (some refined runs do not converge; see res).
"""
import sys
import numpy as np, scipy.sparse as sp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import fsolve
from scipy.sparse.linalg import spsolve, eigs

a = np.array([342, 368, 128, 1024]) / 343
def ph(q): return q * (a[0] + q * (a[1] + q * (a[2] + q * a[3])))
def dph(q): return a[0] + q * (2 * a[1] + q * (3 * a[2] + q * 4 * a[3]))
df = lambda u: -1 + dph(u) / (1 + ph(u)) ** 2
A1, A2, A3, A4 = a.tolist()
def f(u):
    p = u * (A1 + u * (A2 + u * (A3 + u * A4)))
    return -u + p / (1 + p)
s, N, e0, ue = 0.5, 20000, 1e-3, 2e-3
du = (s - e0 - ue) / N
def shoot(c, dense=False):
    m = (-c + (c * c - 4 * df(s)) ** .5) / 2
    E, u = (m * e0) ** 2 / 2, s - e0
    out = [(u, E)]
    G = lambda u, E: -(c * (2 * max(E, 0.)) ** .5 - f(u))
    for k in range(N):
        k1 = G(u, E); k2 = G(u - du / 2, E + du / 2 * k1); k3 = G(u - du / 2, E + du / 2 * k2); k4 = G(u - du, E + du * k3)
        E, u = E + du / 6 * (k1 + 2 * k2 + 2 * k3 + k4), u - du
        if E <= 0:
            return None
        if dense:
            out.append((u, E))
    return np.array(out) if dense else E
def res(c):
    E = shoot(c)
    return 1.0 if E is None else -(2 * E) ** .5 + (c + (c * c - 4 * df(0.)) ** .5) / 2 * ue
lo, hi = 0.0, 0.2
for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (lo, mid) if res(mid) > 0 else (mid, hi)
c = lo
o = shoot(c, True)
ug, wg = o[:, 0], -(2 * np.maximum(o[:, 1], 1e-30)) ** .5
zu = np.concatenate([[0], np.cumsum((ug[1:] - ug[:-1]) * 2 / (wg[1:] + wg[:-1]))])
ms, m0 = (-c + np.sqrt(c * c - 4 * df(s))) / 2, (c + np.sqrt(c * c - 4 * df(0.))) / 2
h = 0.1
Z = np.arange(-200, 700 + h / 2, h / 2)
U = np.interp(Z, zu, ug)
U[Z < zu[0]] = s - e0 * np.exp(ms * (Z[Z < zu[0]] - zu[0]))
U[Z > zu[-1]] = ue * np.exp(-m0 * (Z[Z > zu[-1]] - zu[-1]))
Fp = df(U)
Zc = Z[::2]
w = eigh_tridiagonal(-2 / h ** 2 + df(U[::2]) - c * c / 4, np.ones(len(Zc) - 1) / h ** 2, select="i", select_range=(len(Zc) - 4, len(Zc) - 1))[0]
print("c", round(c, 5), "monotone", bool((np.diff(U) <= 1e-12).all()), "top eigs", np.round(w[::-1], 5), "edge", round(df(0.) - c * c / 4, 5))
def evans(mus):
    mus = np.asarray(mus, dtype=complex)
    rp, rm = -c / 2 - np.sqrt(c * c / 4 + mus - df(0.)), -c / 2 + np.sqrt(c * c / 4 + mus - df(s))
    def run(idx, r, sgn):
        y0, y1, hh = np.ones_like(mus), r.copy(), sgn * h
        G = lambda fp, p0, p1: (p1 - r * p0, -c * p1 - (fp - mus) * p0 - r * p1)
        for k in range(len(idx) // 2):
            f0, fm, f1 = Fp[idx[2 * k]], Fp[idx[2 * k + 1]], Fp[idx[2 * k + 2]]
            k1 = G(f0, y0, y1); k2 = G(fm, y0 + hh / 2 * k1[0], y1 + hh / 2 * k1[1]); k3 = G(fm, y0 + hh / 2 * k2[0], y1 + hh / 2 * k2[1]); k4 = G(f1, y0 + hh * k3[0], y1 + hh * k3[1])
            y0, y1 = y0 + hh / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), y1 + hh / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return y0, y1
    i0 = np.argmin(abs(Z))
    up, um = run(np.arange(len(Z) - 1, i0 - 1, -1), rp, -1), run(np.arange(0, i0 + 1), rm, 1)
    return um[0] * up[1] - um[1] * up[0]
def wind(mus):
    D = evans(mus)
    D = np.append(D, D[0])
    st = np.angle(D[1:] / D[:-1])
    return round(st.sum() / (2 * np.pi), 4), round(abs(st).max(), 3)
for r in (0.0015, 0.0025):
    print("circle", r, "winding, largest angle step", wind(r * np.exp(2j * np.pi * np.arange(400) / 400)))
R, x0 = 2.0, -0.0015
th0 = np.arccos(x0 / R)
arc = R * np.exp(1j * np.linspace(-th0, th0, 800, endpoint=False))
seg = np.r_[1j * np.geomspace(R * np.sin(th0), 1e-4, 800), -1j * np.geomspace(1e-4, R * np.sin(th0), 800)] + x0
print("half-disc winding, largest angle step", wind(np.r_[arc, seg]))

dz, zr = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.05, 40.0)
def study(name, rea, jac, d, behind, ahead, Mt, Dt, trans):
    ls = np.geomspace(1e-3, 30, 6000)
    cc = np.array([np.linalg.eigvals(l * l * Dt + Mt).real.max() for l in ls]) / ls
    i = cc.argmin()
    cs, lst = cc[i], ls[i]
    m = len(d)
    zg = np.arange(-100, zr + dz / 2, dz)
    n = len(zg)
    D2 = sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="lil"); D2[0, 0] = -1; D2 = D2.tocsr() / dz ** 2
    Dc = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="lil"); Dc[0, 0] = -1; Dc[0, 1] = 1; Dc = Dc.tocsr() / (2 * dz)
    Du = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], format="csr") / dz
    for fac in (1.02, 1.1, 1.2):
        cf = fac * cs
        lam = ls[np.where(cc[:i + 1] <= cf)[0].min()]
        W_, V_ = np.linalg.eig(lam * lam * Dt + Mt)
        q = np.abs(V_[:, W_.real.argmax()].real); q = q / q.max()
        L = sp.block_diag([d[k] * D2 + cf * Dc if d[k] > 0 else cf * Du for k in range(m)], format="csr")
        bnd = [n * (k + 1) - 1 for k in range(m)]
        H = 1 / (1 + np.exp(np.clip(lam * zg, -50, 50)))
        u = np.concatenate([ahead[k] + (behind[k] - ahead[k]) * H for k in range(m)])
        ur = ahead.copy(); ur[trans] = ahead[trans] + q * 0.3 * np.exp(-lam * zg[-1])
        for k in range(m):
            u[bnd[k]] = ur[k]
        isb = np.zeros(m * n); isb[bnd] = 1
        dt = 0.5
        for it in range(600):
            Y = u.reshape(m, n)
            F = L @ u + rea(Y).ravel()
            F[bnd] = 0
            A = sp.diags(1 - isb) @ (sp.identity(m * n, format="csr") / dt - (L + jac(Y))) + sp.diags(isb)
            u = u + spsolve(A.tocsc(), F)
            nr = np.abs(F).max(); dt = min(dt * 1.15, 200)
            if nr < 1e-10:
                break
        Y = u.reshape(m, n)
        J = (L + jac(Y)).tocoo()
        ex = np.tile(lst * np.maximum(zg, 0), m)
        Aw = sp.coo_matrix((J.data * np.exp(ex[J.row] - ex[J.col]), (J.row, J.col)), shape=J.shape).tocsc()
        ev = eigs(Aw, k=8, sigma=0.03, which="LM", return_eigenvectors=False)
        ev = ev[np.argsort(-ev.real)]
        ev0 = eigs(J.tocsc(), k=8, sigma=0.03, which="LM", return_eigenvectors=False)
        ev0 = ev0[np.argsort(-ev0.real)]
        print(name, "dz", dz, "zr", zr, "c/c*", fac, "c", round(cf, 4), "res %.0e" % nr, "weighted", np.round(ev[:3], 4), "plain", np.round(ev0[:2], 4), "edge", round(lst * (cs - cf), 4))

mu, Lam, be, xi, ga, de = 0.1, 0.1, 1.0, 1.0, 0.2, 1.0
def rC(Y):
    return np.array([Lam - mu * Y[0] - be * Y[0] * Y[2], be * Y[0] * Y[2] - (mu + ga) * Y[1], xi * Y[1] - de * Y[2]])
def jC(Y):
    S, I, B = Y; e = np.ones(len(S)); g = sp.diags
    return sp.bmat([[g(-mu * e - be * B), None, g(-be * S)], [g(be * B), g(-(mu + ga) * e), g(be * S)], [None, g(xi * e), g(-de * e)]], format="csr")
Ss = (mu + ga) * de / (be * xi); Is = (Lam - mu * Ss) / (mu + ga); Bs = xi * Is / de
EJ = np.array([[-mu - be * Bs, 0, -be * Ss], [be * Bs, -mu - ga, be * Ss], [0, xi, -de]])
print("resident sup_k s(J-k^2 D)", round(max(np.linalg.eigvals(EJ - k * k * np.diag([0, 0, 1.])).real.max() for k in np.linspace(0, 10, 400)), 4))
study("cholera", rC, jC, np.array([0, 0, 1.0]), np.array([Ss, Is, Bs]), np.array([1.0, 0, 0]), np.array([[-(mu + ga), be], [xi, -de]]), np.diag([0, 1.0]), [1, 2])
V = np.array([[1, 0], [-.5, .7]]); Wm = np.array([[.2, .9], [.3, .1]]); Lz = 2.0
def rS(Y):
    q = Wm @ Y
    return -V @ Y + Lz * q / (1 + Lz * q)
def jS(Y):
    q = Wm @ Y; e = Lz / (1 + Lz * q) ** 2
    return sp.bmat([[sp.diags(-V[i, j] * np.ones(q.shape[1]) + e[i] * Wm[i, j]) for j in range(2)] for i in range(2)], format="csr")
xb = fsolve(lambda x: rS(x.reshape(2, 1)).ravel(), [1, 1.])
study("spaFront", rS, jS, np.array([1, .1]), xb, np.zeros(2), Lz * Wm - V, np.diag([1, .1]), [0, 1])
