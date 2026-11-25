"""Task-cycle rotation, part 3 (SPEC3.md): label-space loops.

Fields xi_e, Lie bracket, self-term, stiffness, label-plane tasks and runner,
kernel and CKA measures. Reuses tcr.py / tcr2.py unchanged (imported).
No experiment logic here. All arithmetic float64, CPU, deterministic.
"""
import math

import numpy as np
import torch

import tcr
import tcr2
from tcr import TaskFamily, gn_learn

torch.set_default_dtype(torch.float64)

FD_STEP = 1e-4          # s = FD_STEP / ||w||  (SPEC3 section 1)
END_TOL = 1e-12         # SPEC3 section 7 test 7


# ---------------------------------------------------------------------------
# Section 1: label directions
# ---------------------------------------------------------------------------
def label_dirs(seed, yA_norm, n=tcr.N_TASK):
    """From Generator(555 + seed) draw u_raw then v_raw (n entries ~ N(0,1) each).
    v' = v_raw - (v_raw.u_raw / u_raw.u_raw) u_raw; u, v scaled to norm yA_norm."""
    g = torch.Generator().manual_seed(555 + seed)
    u_raw = torch.randn(n, generator=g)
    v_raw = torch.randn(n, generator=g)
    vp = v_raw - (torch.dot(v_raw, u_raw) / torch.dot(u_raw, u_raw)) * u_raw
    u = u_raw * yA_norm / torch.linalg.norm(u_raw)
    v = vp * yA_norm / torch.linalg.norm(vp)
    return u, v


def random_dirs(gen_seed, count, yA_norm, n=tcr.N_TASK):
    """Sequential draws from Generator(gen_seed): each 16 entries ~ N(0,1), scaled to yA_norm."""
    g = torch.Generator().manual_seed(gen_seed)
    out = []
    for _ in range(count):
        r = torch.randn(n, generator=g)
        out.append(r * yA_norm / torch.linalg.norm(r))
    return out


class Setup:
    """Per (h, seed): family, model, X = X_A, y_A, u, v, theta_init, theta_A."""

    def __init__(self, h, seed, learner="gn", linear=False, deadline=None,
                 uv_from_teacher_for_linear=True):
        self.h, self.seed, self.linear = h, seed, linear
        self.fam = TaskFamily(seed)
        self.X = self.fam.X_A
        self.X_probe = self.fam.X_probe
        if linear:
            self.model = tcr.LinearModel(seed)
            self.theta_init = self.model.init()
            self.yA = self.model.labels(self.X)
            # "same u, v": the label directions of section 1 for this seed. Primary reading:
            # the very same vectors as for the MLP (scaled to the teacher's ||y_A||).
            ynorm = (torch.linalg.norm(self.fam.teacher(self.X)) if uv_from_teacher_for_linear
                     else torch.linalg.norm(self.yA))
        else:
            self.model = tcr2.FastMLP(h)
            self.theta_init = self.model.init(seed)
            self.yA = self.fam.teacher(self.X)
            ynorm = torch.linalg.norm(self.yA)
        self.yA_norm = ynorm
        self.u, self.v = label_dirs(seed, ynorm)
        self.A = (self.X, self.yA)
        if learner == "gn":
            self.theta_A, self.base_info = gn_learn(self.model, self.theta_init, self.A)
        else:
            self.theta_A, self.base_info = tcr.gd_learn(self.model, self.theta_init, self.A,
                                                        deadline=deadline)

    def task(self, y):
        return (self.X, y)


# ---------------------------------------------------------------------------
# Kernel, fields, bracket, self-term, stiffness
# ---------------------------------------------------------------------------
def jk(model, theta, X):
    J = model.jac(theta, X)
    return J, J @ J.T


def xi(model, theta, X, E, JK=None):
    """xi_E(theta) = J^T K^{-1} E. E: (n,) -> (P,), or (n,k) -> (P,k)."""
    J, K = JK if JK is not None else jk(model, theta, X)
    return J.T @ torch.linalg.solve(K, E)


def dxi(model, theta, X, E, w, step=FD_STEP):
    """Central FD directional derivative D xi_E . w, s = step / ||w||."""
    s = step / torch.linalg.norm(w).item()
    return (xi(model, theta + s * w, X, E) - xi(model, theta - s * w, X, E)) / (2 * s)


def proj_v(J, K, w):
    """P_v w = w - J^T K^{-1} J w."""
    return w - J.T @ torch.linalg.solve(K, J @ w)


def bracket(model, theta, X, e1, e2, step=FD_STEP):
    """b_{e1 e2} = D xi_{e2} . xi_{e1} - D xi_{e1} . xi_{e2}   (Lie bracket [xi_e1, xi_e2])."""
    JK = jk(model, theta, X)
    x1 = xi(model, theta, X, e1, JK)
    x2 = xi(model, theta, X, e2, JK)
    return dxi(model, theta, X, e2, x1, step) - dxi(model, theta, X, e1, x2, step)


def self_term(model, theta, X, e, step=FD_STEP):
    """c_e = P_v [ D xi_e . xi_e ]."""
    J, K = jk(model, theta, X)
    x = xi(model, theta, X, e, (J, K))
    return proj_v(J, K, dxi(model, theta, X, e, x, step))


def stiffness(model, theta, X, e):
    """phi_e = 1/2 e^T K^{-1} e."""
    _, K = jk(model, theta, X)
    return 0.5 * torch.dot(e, torch.linalg.solve(K, e)).item()


def kernel_stats(model, theta, X, u=None, v=None):
    J, K = jk(model, theta, X)
    lam = torch.linalg.eigvalsh(K)
    out = {"trKinv": torch.trace(torch.linalg.inv(K)).item(),
           "lam_min": lam[0].item(), "lam_max": lam[-1].item()}
    if u is not None:
        out["phi_u"] = 0.5 * torch.dot(u, torch.linalg.solve(K, u)).item()
    if v is not None:
        out["phi_v"] = 0.5 * torch.dot(v, torch.linalg.solve(K, v)).item()
    return out


def curvature_map(model, theta, X, step=FD_STEP):
    """L6. Eigen-decomposition K = sum lam_i q_i q_i^T (descending). Returns lam (16,),
    B (16x16, B_ij = ||b_{q_i q_j}||, symmetric, diag 0), kappa (B_ij / (||xi_i|| ||xi_j||)),
    kappa_alt (B_ij sqrt(lam_i lam_j)).
    D xi_{q_j} . xi_{q_i} is computed for every ordered pair: for each direction w = xi_{q_i}
    (step s = step/||w||) all 16 fields xi_{q_j} are evaluated at theta +- s w."""
    J, K = jk(model, theta, X)
    lam, Q = torch.linalg.eigh(K)
    idx = torch.argsort(lam, descending=True)
    lam, Q = lam[idx], Q[:, idx]
    n = Q.shape[1]
    XI = J.T @ torch.linalg.solve(K, Q)                 # (P, n): column i = xi_{q_i}
    # Dmat[i][:, j] = D xi_{q_j} . xi_{q_i}
    Dmat = []
    for i in range(n):
        w = XI[:, i]
        s = step / torch.linalg.norm(w).item()
        Dmat.append((xi(model, theta + s * w, X, Q) - xi(model, theta - s * w, X, Q)) / (2 * s))
    B = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            b = Dmat[i][:, j] - Dmat[j][:, i]          # D xi_qj . xi_qi - D xi_qi . xi_qj
            B[i, j] = B[j, i] = torch.linalg.norm(b)
    xin = torch.linalg.norm(XI, dim=0)
    kappa = B / (xin[:, None] * xin[None, :])
    kappa_alt = B * torch.sqrt(lam[:, None] * lam[None, :])
    kappa.fill_diagonal_(0.0)
    kappa_alt.fill_diagonal_(0.0)
    return lam, B, kappa, kappa_alt


# ---------------------------------------------------------------------------
# Feature measures
# ---------------------------------------------------------------------------
def last_hidden(model, theta, X):
    """tanh(z2) on X: (m, h)."""
    p = model.unflat(theta)
    a1 = torch.tanh(X @ p["W1"].T / model.sd + p["b1"])
    return torch.tanh(a1 @ p["W2"].T / model.sh + p["b2"])


def linear_cka(A, B):
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    num = torch.linalg.norm(B.T @ A) ** 2
    den = torch.linalg.norm(A.T @ A) * torch.linalg.norm(B.T @ B)
    return (num / den).item()


# ---------------------------------------------------------------------------
# Label plane and runners
# ---------------------------------------------------------------------------
class LabelPlane:
    """Task at plane point (x,y): (X_A, y_A + eps (x u + y v)); (0,0) is task A itself."""

    def __init__(self, S, eps):
        self.S, self.eps = S, eps
        self.A = S.A

    def __call__(self, p):
        p = (float(p[0]), float(p[1]))
        if p == (0.0, 0.0):
            return self.A
        return (self.S.X, self.S.yA + self.eps * (p[0] * self.S.u + p[1] * self.S.v))


class MixedPlane:
    """Part M: (1,0) -> inputs move (X_A + eps U, g(X_A + eps U)); (0,1) -> labels move
    (X_A, y_A + eps v); (0,0) -> A."""

    def __init__(self, S, eps):
        self.S, self.eps = S, eps
        self.A = S.A
        XB = S.fam.X_A + eps * S.fam.U
        self.B = (XB, S.fam.teacher(XB))
        self.C = (S.X, S.yA + eps * S.v)

    def __call__(self, p):
        p = (float(p[0]), float(p[1]))
        if p == (0.0, 0.0):
            return self.A
        if p == (1.0, 0.0):
            return self.B
        if p == (0.0, 1.0):
            return self.C
        raise ValueError(p)


def end_check(model, theta, task, tol=END_TOL):
    """SPEC3 section 7 test 7: ||f(X;theta) - y||_inf <= tol at the end of every sequence."""
    X, y = task
    r = (model.f(theta, X) - y).abs().max().item()
    assert r <= tol, f"sequence end residual {r:.3e} > {tol:.0e}"
    return r


def poly_run(S, learn, theta_A, pts, plane, m_list=(1,), check=True):
    """Forward/reverse polygon runner (tcr2.polygon_run) with the end-of-sequence check."""
    res, ip, im, thp, thm = tcr2.polygon_run(S.model, learn, theta_A, pts, plane,
                                             S.X_probe, m_list=m_list)
    ends = []
    if check:
        for m in m_list:
            ends.append(end_check(S.model, thp[m], S.A))
            ends.append(end_check(S.model, thm[m], S.A))
    return res, ip + im, thp, thm, (max(ends) if ends else None)


def out_and_back(S, learn, theta_A, e, eps, m=1, record=None, check=True):
    """Repeat m times: learn (X, y_A + eps e), learn A. record: set of m at which to
    store theta. Returns (theta_final, {m: theta}, infos)."""
    record = set(record or [m])
    th = theta_A
    out, infos = {}, []
    Te = (S.X, S.yA + eps * e)
    for k in range(1, m + 1):
        th, i1 = learn(th, Te)
        th, i2 = learn(th, S.A)
        infos += [i1, i2]
        if check:
            end_check(S.model, th, S.A)
        if k in record:
            out[k] = th.clone()
    return th, out, infos


def random_excursions(S, learn, theta_A, dirs, eps, record, check=True):
    """Excursion k uses dirs[k-1]: learn (X, y_A + eps e_k), learn A."""
    th = theta_A
    out, infos = {}, []
    for k, e in enumerate(dirs, start=1):
        th, i1 = learn(th, (S.X, S.yA + eps * e))
        th, i2 = learn(th, S.A)
        infos += [i1, i2]
        if check:
            end_check(S.model, th, S.A)
        if k in record:
            out[k] = th.clone()
    return th, out, infos


def refine_path(S, learn, theta_A, e, L, N, check=True):
    """K4: learn (X, y_A + (kL/N) e) for k = 1..N. Endpoint check is against the final
    target y_A + L e (the sequence does not end in A)."""
    th = theta_A
    infos = []
    for k in range(1, N + 1):
        th, i = learn(th, (S.X, S.yA + (k * L / N) * e))
        infos.append(i)
    if check:
        end_check(S.model, th, (S.X, S.yA + L * e))
    return th, infos


def nrm(v):
    return torch.linalg.norm(v).item()


def cos(a, b):
    return tcr.cosine(a, b)


def loglog_slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])
