"""Task-cycle rotation, part 7 (SPEC7.md): where does the rise in sharpness come from?

Machinery for Tests 10-12: fast numpy model wrapper (outputs, gradient, Jacobian, kernel),
nearest-point learner NP (SPEC7 1.1), straight-line transport SL (1.2), GN / GD learners with
C and Q bookkeeping, Hessian sharpness S_H (eigsh on central-difference HVPs), and the
prediction pred(C; theta_b) = D lambda_1[LAW_theta_b(C)].
Reuses tcr.py / tcr3.py / tcr4.py / tcr6.py unchanged (imported). No grid logic here.
float64, CPU, deterministic. Thread count fixed to 2 by tcr4 (imported first).
"""
import tcr4  # noqa: F401  (first: fixes thread count before numpy / torch are used)
import math
import time

import numpy as np
import scipy.sparse.linalg as spla
import torch

import tcr
import tcr6
from tcr4 import Law  # noqa: F401

torch.set_default_dtype(torch.float64)

N = 16
NP_MAXIT = 200
NP_TOL_RES = 1e-10
NP_TOL_STAT = 1e-8
NP_MERIT_W = 1e-6
NP_MIN_STEP = 2.0 ** -20
NP_CONT = 8
SL_M = 64
SH_DELTA = 1e-5
SH_TOL = 1e-10          # eigsh tolerance (not fixed by the spec; ARPACK's default 0 = machine
                        # precision is below the FD noise of the HVP; see U4)
SH_V0_SEED = 4242


def phi(c):
    """phi(c) = (1 - c)/(2 - c)  (SPEC7 U6 / Test 10 closed form)."""
    return (1.0 - c) / (2.0 - c)


# ---------------------------------------------------------------------------
# numpy model wrapper
# ---------------------------------------------------------------------------
class Net:
    """Fast numpy evaluation of F, J^T r / n, J, K = J J^T for tcr2.FastMLP / tcr.LinearModel."""

    def __init__(self, model, X):
        self.model = model
        self.X = X
        self.n = X.shape[0]
        self.P = model.P
        self.linear = isinstance(model, tcr.LinearModel)
        if self.linear:
            self.Phi = model.phi(X).numpy().copy()
            self.K_lin = self.Phi @ self.Phi.T
        else:
            self.m = tcr6.NpMLP6(model, X)
            Xn = self.m.Xn
            self.XX = Xn @ Xn.T / model.d + 1.0

    # --- MLP internals
    def _acts(self, t):
        m = self.m
        W1, b1, W2, b2, w3 = (m._v(t, k) for k in ("W1", "b1", "W2", "b2", "w3"))
        a1 = np.tanh(m.Xn @ W1.T / m.sd + b1)
        a2 = np.tanh(a1 @ W2.T / m.sh + b2)
        return W2, w3, a1, a2

    def F(self, t):
        if self.linear:
            return self.Phi @ t
        return self.m.fwd(t)[0]

    def fg(self, t, y):
        """F, r = F - y, g = J^T r / n."""
        if self.linear:
            F = self.Phi @ t
            r = F - y
            return F, r, self.Phi.T @ r / self.n
        F, acts = self.m.fwd(t)
        r = F - y
        return F, r, self.m.vjp(t, acts, r / self.n)

    def grad(self, t, y):
        return self.fg(t, y)[2]

    def jac(self, t):
        if self.linear:
            return self.Phi.copy()
        m = self.m
        W2, w3, a1, a2 = self._acts(t)
        n, h, d = self.n, m.sh ** 2, m.sd ** 2
        h, d = int(round(h)), int(round(d))
        g2 = w3[None, :] * (1.0 - a2 * a2) / m.sh
        g1 = (g2 @ W2 / m.sh) * (1.0 - a1 * a1)
        J = np.empty((n, self.P))
        sl = m.sl
        J[:, sl["W1"][0]:sl["W1"][1]] = (g1[:, :, None] * m.Xn[:, None, :] / m.sd).reshape(n, h * d)
        J[:, sl["b1"][0]:sl["b1"][1]] = g1
        J[:, sl["W2"][0]:sl["W2"][1]] = (g2[:, :, None] * a1[:, None, :] / m.sh).reshape(n, h * h)
        J[:, sl["b2"][0]:sl["b2"][1]] = g2
        J[:, sl["w3"][0]:sl["w3"][1]] = a2 / m.sh
        J[:, m.b3i] = 1.0
        return J

    def K(self, t):
        """K = J J^T by the closed form (g1 g1^T)o(XX^T/d + 1) + (g2 g2^T)o(a1 a1^T/h + 1)
        + a2 a2^T/h + 1 (checked against J J^T and tcr2.FastMLP.jac in test_part7.py)."""
        if self.linear:
            return self.K_lin.copy()
        m = self.m
        W2, w3, a1, a2 = self._acts(t)
        h = m.sh ** 2
        g2 = w3[None, :] * (1.0 - a2 * a2) / m.sh
        g1 = (g2 @ W2 / m.sh) * (1.0 - a1 * a1)
        return ((g1 @ g1.T) * self.XX + (g2 @ g2.T) * (a1 @ a1.T / h + 1.0)
                + a2 @ a2.T / h + 1.0)

    def kstats(self, t):
        K = self.K(t)
        ev = np.linalg.eigvalsh(K)
        return {"lam1": float(ev[-1]), "lam_min": float(ev[0]), "trK": float(np.trace(K))}

    def lam1(self, t):
        return float(np.linalg.eigvalsh(self.K(t))[-1])


def tnp(x):
    return x.numpy().copy() if isinstance(x, torch.Tensor) else np.asarray(x, float)


# ---------------------------------------------------------------------------
# Hessian sharpness S_H
# ---------------------------------------------------------------------------
def hvp_fn(net, th, y, delta=SH_DELTA):
    def mv(v):
        v = np.asarray(v, float).ravel()
        nv = float(np.linalg.norm(v))
        if nv == 0.0:
            return np.zeros_like(v)
        vh = v / nv
        return (net.grad(th + delta * vh, y) - net.grad(th - delta * vh, y)) * (nv / (2 * delta))
    return mv


def sharp_H(net, th, y, delta=SH_DELTA, tol=SH_TOL):
    """Largest eigenvalue of the Hessian of L = (1/2n)||F - y||^2 at th (eigsh, which='LA',
    fixed start vector for determinism)."""
    th = tnp(th)
    y = tnp(y)
    P = th.shape[0]
    op = spla.LinearOperator((P, P), matvec=hvp_fn(net, th, y, delta), dtype=np.float64)
    v0 = np.random.default_rng(SH_V0_SEED).standard_normal(P)
    val = spla.eigsh(op, k=1, which="LA", v0=v0, tol=tol, return_eigenvectors=False)
    return float(val[0])


def hessian_dense(net, th, y, delta=SH_DELTA):
    th, y = tnp(th), tnp(y)
    P = th.shape[0]
    mv = hvp_fn(net, th, y, delta)
    H = np.empty((P, P))
    e = np.zeros(P)
    for j in range(P):
        e[j] = 1.0
        H[:, j] = mv(e)
        e[j] = 0.0
    return H


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def identity_err(C, Q, dY):
    """relative error of C + C^T = dY dY^T - Q."""
    lhs = C + C.T
    rhs = np.outer(dY, dY) - Q
    den = max(np.linalg.norm(rhs), np.linalg.norm(lhs), 1e-300)
    return float(np.linalg.norm(lhs - rhs) / den)


def pieces(C, Q, dY):
    """travel = 1/2 dY dY^T, jitter = -1/2 Q, area = 1/2 (C - C^T)."""
    return 0.5 * np.outer(dY, dY), -0.5 * Q, 0.5 * (C - C.T)


# ---------------------------------------------------------------------------
# GD with C, Q bookkeeping (same arithmetic as tcr4.fo_learn_C(kind='GD'))
# ---------------------------------------------------------------------------
def gd_run(net, th0, y, eta, tol=tcr4.FO_TOL, cap=tcr4.FO_CAP, div=tcr4.DIV_FACTOR,
           after_step=None):
    """theta <- theta - eta J^T r / n until ||r||_inf <= tol / cap / DIVERGED.
    C += (F_{k+1}-F_k) (x) (F_k - F_start); Q += (F_{k+1}-F_k)(F_{k+1}-F_k)^T.
    after_step(k, th, F, r, C, Q): called after step k (k = 1, 2, ...) with the new state;
    may return True to request a stop (status STOPPED)."""
    t0 = time.time()
    th = tnp(th0)
    y = tnp(y)
    F, r, g = net.fg(th, y)
    Fs = F.copy()
    n = F.shape[0]
    C = np.zeros((n, n))
    Q = np.zeros((n, n))
    r0 = float(np.abs(r).max())
    k = 0
    while True:
        res = float(np.abs(r).max())
        if not np.isfinite(res) or res > div * r0:
            status = "DIVERGED"
            break
        if res <= tol:
            status = "CONVERGED"
            break
        if k >= cap:
            status = "NOT CONVERGED"
            break
        dth = -eta * g
        th = th + dth
        Fo = F
        F, r, g = net.fg(th, y)
        dF = F - Fo
        C += np.outer(dF, Fo - Fs)
        Q += np.outer(dF, dF)
        k += 1
        if after_step is not None and after_step(k, th, F, r, C, Q):
            status = "STOPPED"
            break
    return th, C, Q, F - Fs, {"steps": k, "res_inf": float(np.abs(r).max()), "res0_inf": r0,
                              "status": status, "eta": eta, "time": time.time() - t0}


def eta_c(model, theta, X, c):
    """eta = c n / lambda_1(K(theta_in))  (tcr4 convention, same lam_max_K routine)."""
    return c * X.shape[0] / tcr4.lam_max_K(model, theta if isinstance(theta, torch.Tensor)
                                           else torch.from_numpy(theta), X)


# ---------------------------------------------------------------------------
# GN with C, Q bookkeeping (tcr.gn_step, i.e. identical to tcr4.gn_learn_C + Q)
# ---------------------------------------------------------------------------
def gn_CQ(model, theta, X, y, F_ref, C, Q, tol=tcr.GN_TOL, max_iter=tcr.GN_MAX_ITER):
    F = model.f(theta, X)
    r = F - y
    it, n_lsf = 0, 0
    while r.abs().max().item() > tol:
        if it >= max_iter:
            raise tcr.NotConverged(f"GN not converged after {max_iter} iterations, "
                                   f"||r||_inf={r.abs().max().item():.3e}")
        theta, r, s, lsf = tcr.gn_step(model, theta, X, y)
        F_new = model.f(theta, X)
        dF = (F_new - F).numpy()
        C += np.outer(dF, F.numpy() - F_ref)
        Q += np.outer(dF, dF)
        F = F_new
        n_lsf += int(lsf)
        it += 1
    return theta, {"steps": it, "res_inf": r.abs().max().item(), "ls_fail": n_lsf}


def gn_run(model, theta, X, y):
    """GN learn from theta to y. Returns theta, C, Q, dY, info (status CONVERGED / GN-FAILED)."""
    t0 = time.time()
    F_ref = model.f(theta, X).numpy().copy()
    C = np.zeros((N, N))
    Q = np.zeros((N, N))
    try:
        th, info = gn_CQ(model, theta, X, y, F_ref, C, Q)
        info["status"] = "CONVERGED"
    except tcr.NotConverged as e:
        return None, C, Q, None, {"status": "GN-FAILED", "error": str(e), "time": time.time() - t0}
    dY = model.f(th, X).numpy() - F_ref
    info["time"] = time.time() - t0
    return th, C, Q, dY, info


def sl_run(model, theta_b, X, t, m=SL_M):
    """SL(theta_b, t; m): sub-targets F(theta_b) + (j/m)(t - F(theta_b)), j = 1..m, each fitted by
    the GN learner from the previous point. C, Q accumulated over all GN iterations (relative to
    F(theta_b))."""
    t0 = time.time()
    t = torch.as_tensor(t)
    Fb = model.f(theta_b, X)
    F_ref = Fb.numpy().copy()
    C = np.zeros((N, N))
    Q = np.zeros((N, N))
    th = theta_b
    steps, lsf = 0, 0
    for j in range(1, m + 1):
        target = Fb + (j / m) * (t - Fb)
        try:
            th, info = gn_CQ(model, th, X, target, F_ref, C, Q)
        except tcr.NotConverged as e:
            return None, C, Q, None, {"status": "SL-FAILED", "error": str(e), "sub": j,
                                      "time": time.time() - t0}
        steps += info["steps"]
        lsf += info["ls_fail"]
    dY = model.f(th, X).numpy() - F_ref
    return th, C, Q, dY, {"status": "CONVERGED", "steps": steps, "ls_fail": lsf,
                          "res_inf": float((model.f(th, X) - t).abs().max().item()),
                          "time": time.time() - t0}


# ---------------------------------------------------------------------------
# NP: nearest point with outputs t (SPEC7 1.1)
# ---------------------------------------------------------------------------
def _np_core(net, thb, th, t, max_iter=NP_MAXIT, tol_res=NP_TOL_RES, tol_stat=NP_TOL_STAT,
             merit="merit"):
    def state(x):
        F = net.F(x)
        J = net.jac(x)
        K = J @ J.T
        dth = x - thb
        Pv = dth - J.T @ np.linalg.solve(K, J @ dth)
        return F, J, K, dth, Pv

    def M(F, Pv):
        if merit == "merit":
            return float(np.dot(F - t, F - t) + NP_MERIT_W * np.dot(Pv, Pv))
        return float(np.dot(F - t, F - t))

    F, J, K, dth, Pv = state(th)
    ls_fail = 0
    it = 0
    while True:
        res = float(np.abs(F - t).max())
        nd = float(np.linalg.norm(dth))
        stat = float(np.linalg.norm(Pv)) / max(nd, 1e-12)
        if res <= tol_res and stat <= tol_stat:
            return th, {"converged": True, "iters": it, "res_inf": res, "stat": stat,
                        "dist": nd, "ls_fail": ls_fail}
        if it >= max_iter or not np.isfinite(res):
            return th, {"converged": False, "iters": it, "res_inf": res, "stat": stat,
                        "dist": nd, "ls_fail": ls_fail}
        d = thb + J.T @ np.linalg.solve(K, (t - F) + J @ dth) - th
        M0 = M(F, Pv)
        s = 1.0
        while True:
            x = th + s * d
            st = state(x)
            if M(st[0], st[4]) < M0:
                break
            if s <= NP_MIN_STEP:
                ls_fail += 1
                break          # literal: take the smallest step anyway
            s *= 0.5
        th = x
        F, J, K, dth, Pv = st
        it += 1


def np_solve(net, theta_b, t, merit="merit"):
    """NP(theta_b, t). Status CONVERGED, CONVERGED-CONT (after the continuation fallback) or
    NP-FAILED. Returns (theta or None, info)."""
    t0 = time.time()
    thb = tnp(theta_b)
    t = tnp(t)
    th, info = _np_core(net, thb, thb.copy(), t, merit=merit)
    info = dict(info)
    if info["converged"]:
        info["status"] = "CONVERGED"
        info["time"] = time.time() - t0
        return th, info
    first = dict(info)
    Fb = net.F(thb)
    x = thb.copy()
    conts = []
    ok = True
    for j in range(1, NP_CONT + 1):
        tj = Fb + (j / NP_CONT) * (t - Fb)
        x, ij = _np_core(net, thb, x, tj, merit=merit)
        conts.append(ij)
        if not ij["converged"] and j == NP_CONT:
            ok = False
    last = conts[-1]
    out = {"first_attempt": first, "cont": conts, "iters": first["iters"] + sum(c["iters"] for c in conts),
           "res_inf": last["res_inf"], "stat": last["stat"], "dist": last["dist"],
           "time": time.time() - t0}
    if ok and last["converged"]:
        out["status"] = "CONVERGED-CONT"
        return x, out
    out["status"] = "NP-FAILED"
    return None, out


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
class Pred:
    """pred(C; theta_b) = D lambda_1[LAW_theta_b(C)] (tcr6.dderiv of tcr6.lam1, FD step 1e-4)."""

    def __init__(self, model, theta_b, X, L=None):
        self.model, self.theta_b, self.X = model, theta_b, X
        self.L = L if L is not None else Law(model, theta_b, X)
        self.f = lambda th: tcr6.lam1(model, th, X)  # noqa: E731

    def V(self, C):
        return self.L(C)

    def __call__(self, C):
        return tcr6.dderiv(self.f, self.theta_b, self.L(C))

    def dir(self, V):
        return tcr6.dderiv(self.f, self.theta_b, V)

    def dtr(self, V):
        g = lambda th: torch.trace(tcr6.K_of(self.model, th, self.X)).item()  # noqa: E731
        return tcr6.dderiv(g, self.theta_b, V)


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.polyfit(x, y, 1)[0])


def nrm(v):
    return float(np.linalg.norm(v))


def safe(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
