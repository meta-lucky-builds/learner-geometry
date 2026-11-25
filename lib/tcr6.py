"""Task-cycle rotation, part 6 (SPEC6.md): five CPU tests of the three-force picture.

Objects and per-test machinery for Tests 5-9. Reuses tcr.py / tcr2.py / tcr3.py / tcr4.py
unchanged (imported). No experiment-grid logic here (that is run6.py).
All arithmetic float64, CPU, deterministic. Thread count fixed to 2 by tcr4 (imported first:
it sets OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=2 before numpy is imported and calls
torch.set_num_threads(2)).
"""
import tcr4  # noqa: F401  (first: fixes thread count before numpy / torch are used)
import math
import time

import numpy as np
import scipy.linalg
import torch

import tcr
import tcr3
from tcr3 import xi, jk, proj_v, nrm, cos
from tcr4 import Law, c_parts, fo_learn_C, gn_learn_C, np_model, NotConverged  # noqa: F401

torch.set_default_dtype(torch.float64)

FD_STEP = 1e-4            # "D lambda[V]": central FD, step 1e-4/||V||, times ||V||   (SPEC6 sec. 0)
N = tcr.N_TASK            # 16

# ---------------------------------------------------------------------------
# Shared objects (SPEC6 section 0)
# ---------------------------------------------------------------------------


def setup(h, seed, linear=False):
    """tcr4.setup (= SPEC3 section 1: family, model, X = X_A, y_A, u, v, theta_A by GN) plus the
    eigen-decomposition of K(theta_A) in descending order."""
    S = tcr4.setup(h, seed, linear=linear)
    S.J, S.K = jk(S.model, S.theta_A, S.X)
    S.lam, S.Q = eig_desc(S.K)
    S.uhat = S.u / torch.linalg.norm(S.u)
    S.vhat = S.v / torch.linalg.norm(S.v)
    S._law = None
    return S


def law(S):
    if S._law is None:
        S._law = Law(S.model, S.theta_A, S.X)
    return S._law


def eig_desc(K):
    lam, Q = torch.linalg.eigh(K)
    idx = torch.argsort(lam, descending=True)
    return lam[idx], Q[:, idx]


def K_of(model, theta, X):
    J = model.jac(theta, X)
    return J @ J.T


def lam1(model, theta, X):
    return torch.linalg.eigvalsh(K_of(model, theta, X))[-1].item()


def dderiv(fn, theta, V, step=FD_STEP):
    """D fn[V]: central FD along V/||V|| with step `step`, multiplied by ||V||
    (= (fn(theta + sV) - fn(theta - sV)) / (2s), s = step/||V||)."""
    nv = nrm(V)
    if nv == 0:
        return 0.0
    s = step / nv
    return (fn(theta + s * V) - fn(theta - s * V)) / (2 * s)


def T_matrix(L):
    """(P, n*n) matrix whose column i*n + j is T_ij = P_v[D xi_{e_i} . xi_{e_j}](theta_A).
    tcr4.Law stores L.T[j][:, i] = D xi_{e_i} . xi_{e_j} (unprojected)."""
    n = L.K.shape[0]
    raw = L.T.permute(1, 2, 0).reshape(L.T.shape[1], n * n)   # [p, i*n + j] = L.T[j, p, i]
    return raw - L.J.T @ torch.linalg.solve(L.K, L.J @ raw)


def T_ij(L, i, j):
    return L.Pv(L.T[j][:, i])


def grad_lam1_in_T_span(S, rank_tol=1e-10):
    """g = P_v grad lambda_1 at theta_A, obtained as the projection of grad lambda_1 onto the span
    of {T_ij}: orthonormal basis b_k of span{T_ij} by SVD of the (P x 256) matrix (singular values
    > rank_tol * s_max kept), component c_k = D lambda_1[b_k] by central FD (step 1e-4), g = sum c_k b_k.
    Returns dict with g, G = D lambda_1[g] (the SPEC6 definition of ||P_v grad lambda_1||^2),
    sumsq = sum c_k^2 (same quantity, without the extra FD), rank, and the cross-check against the
    identity P_v grad lambda_1 = 2 lambda_1^2 LAW(q_1 q_1^T)."""
    L = law(S)
    Tm = T_matrix(L)
    U, sv, _ = torch.linalg.svd(Tm, full_matrices=False)
    keep = sv > rank_tol * sv[0]
    B = U[:, keep]
    f = lambda th: lam1(S.model, th, S.X)  # noqa: E731
    c = torch.tensor([dderiv(f, S.theta_A, B[:, k]) for k in range(B.shape[1])])
    g = B @ c
    G = dderiv(f, S.theta_A, g)
    q1 = S.Q[:, 0].numpy()
    g_id = 2.0 * S.lam[0].item() ** 2 * L(np.outer(q1, q1))
    return {"g": g, "G": G, "sumsq": float((c * c).sum()), "rank": int(keep.sum()),
            "sv_max": sv[0].item(), "sv_min_kept": sv[keep][-1].item(),
            "sv_ratio_last": (sv[-1] / sv[0]).item(),
            "cos_g_identity": cos(g, g_id), "ratio_g_identity": nrm(g) / nrm(g_id),
            "G_identity": dderiv(f, S.theta_A, g_id)}


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------
def gd_C(S, y, c, theta=None, fixed_steps=None, hook=None):
    """GD(c) learn from theta (default theta_A) to target y with measured C (SPEC4 rule)."""
    npm = np_model(S.model, S.X)
    th0 = S.theta_A if theta is None else theta
    C = np.zeros((N, N))
    return fo_learn_C(npm, S.model, th0, S.X, y.numpy(), S.F_A_np, C, "GD", c,
                      fixed_steps=fixed_steps, hook=hook)


def gn_C(S, y, theta=None):
    th0 = S.theta_A if theta is None else theta
    return gn_learn_C(S.model, th0, S.X, y, S.F_A_np, np.zeros((N, N)))


def pred2_factor(c):
    return (1.0 - c) / (2.0 * (2.0 - c))


def protocol_B(S, y, c, n_steps=3000, blowup=1e6):
    """GD with eta = c n / lambda_max(K(theta_A)) toward fixed target y for n_steps steps.
    Records per step k (before the step): lambda_1(K(theta_k)), x_k = q_1(theta_k).r_k, loss,
    c_eff = eta lambda_1 / n. Stops early only if ||r|| > blowup."""
    n = N
    eta = c * n / S.lam[0].item()
    th = S.theta_A.clone()
    lamk, xk, loss, ceff, rnorm = [], [], [], [], []
    stopped = None
    for k in range(n_steps + 1):
        J = S.model.jac(th, S.X)
        K = J @ J.T
        lam, Q = torch.linalg.eigh(K)
        F = S.model.f(th, S.X)
        r = F - y
        lamk.append(lam[-1].item())
        xk.append(torch.dot(Q[:, -1], r).item())
        loss.append(0.5 * torch.dot(r, r).item() / n)
        ceff.append(eta * lam[-1].item() / n)
        rnorm.append(nrm(r))
        if not math.isfinite(rnorm[-1]) or rnorm[-1] > blowup:
            stopped = k
            break
        if k == n_steps:
            break
        th = th - eta * (J.T @ r) / n
    return {"lam1": lamk, "x": xk, "loss": loss, "c_eff": ceff, "rnorm": rnorm,
            "stopped_at": stopped, "eta": eta}


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    Xd = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------
def gn_learn_fn(S):
    return lambda th, task: tcr.gn_learn(S.model, th, task)


def gd1_learn_fn(S):
    """GD(1) of SPEC6/SPEC4 = tcr.gd_learn (eta = 1 n / lambda_max(K(theta_in)), tol 1e-8, cap 4e5)."""
    return lambda th, task: tcr.gd_learn(S.model, th, task)


def practice_states(S, e, learner, eps=0.1, ms=(4, 8, 16)):
    """theta_A after m out-and-back excursions along e (targets y_A + eps e, then y_A)."""
    learn = gn_learn_fn(S) if learner == "GN" else gd1_learn_fn(S)
    th, out, infos = tcr3.out_and_back(S, learn, S.theta_A, e, eps, m=max(ms), record=set(ms),
                                       check=(learner == "GN"))
    return out, infos


def gd_fixed(npm, theta, y, eta, k):
    """Exactly k plain GD steps theta <- theta - eta J^T r / n. Returns (F_end, r_0, r_end)."""
    th = theta.numpy().copy()
    F, r, g, _ = npm.fg(th, y)
    r0 = r.copy()
    for _ in range(k):
        th -= eta * g
        F, r, g, _ = npm.fg(th, y)
    return F, r0, r


def interference_probe(S, theta, bdirs, eta0, delta, k=50):
    """SPEC6 Test 6 probe from state theta."""
    npm = np_model(S.model, S.X)
    yA = S.yA.numpy()
    uh, vh = S.uhat.numpy(), S.vhat.numpy()
    nu_u, nu_v, pis = [], [], []
    for b in bdirs:
        F, r0, r = gd_fixed(npm, theta, yA + delta * b, eta0, k)
        d = F - yA
        nu_u.append(float(uh @ d) ** 2 / delta ** 2)
        nu_v.append(float(vh @ d) ** 2 / delta ** 2)
        pis.append(1.0 - np.linalg.norm(r) / np.linalg.norm(r0))
    out = {"nu_u": float(np.mean(nu_u)), "nu_v": float(np.mean(nu_v)),
           "pi_generic": float(np.mean(pis))}
    for tag, e in (("u", uh), ("v", vh)):
        F, r0, r = gd_fixed(npm, theta, yA + delta * e, eta0, k)
        out["pi_" + tag] = 1.0 - float(np.linalg.norm(r) / np.linalg.norm(r0))
    # linear predictions with K(S)
    K = K_of(S.model, theta, S.X).numpy()
    lam, Qm = np.linalg.eigh(K)
    Am = (Qm * (1.0 - eta0 / N * lam) ** k) @ Qm.T
    IA = np.eye(N) - Am
    B = np.stack(bdirs)
    for tag, e in (("u", uh), ("v", vh)):
        out[f"nu_{tag}_lin"] = float(np.linalg.norm(IA @ e) ** 2 / N)
        out[f"pi_{tag}_lin"] = 1.0 - float(np.linalg.norm(Am @ e))
        # supplementary: linear prediction evaluated on the same 16 b_j (sample-matched)
        out[f"nu_{tag}_lin_sample"] = float(np.mean((B @ (IA @ e)) ** 2))
    out["end_res_inf"] = float(np.abs(S.model.f(theta, S.X).numpy() - yA).max())
    return out


def unit_dirs(gen_seed, count, n=N):
    g = torch.Generator().manual_seed(gen_seed)
    out = []
    for _ in range(count):
        r = torch.randn(n, generator=g)
        out.append((r / torch.linalg.norm(r)).numpy())
    return out


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------
def strong_weak_dirs(S):
    """Generator(7000+seed): coefficient draws in the order s (4), w (4), s' (4), w' (4), each
    torch.randn(4). s in span{q_1..q_4}, w in span{q_13..q_16}; s' orthogonalised against s,
    w' against w; all normalised and scaled to ||y_A||."""
    g = torch.Generator().manual_seed(7000 + S.seed)
    cs, cw, cs2, cw2 = (torch.randn(4, generator=g) for _ in range(4))
    Qs, Qw = S.Q[:, :4], S.Q[:, 12:16]
    yn = torch.linalg.norm(S.yA)

    def unit(x):
        return x / torch.linalg.norm(x)
    s = unit(Qs @ cs)
    w = unit(Qw @ cw)
    s2 = Qs @ cs2
    s2 = unit(s2 - torch.dot(s2, s) * s)
    w2 = Qw @ cw2
    w2 = unit(w2 - torch.dot(w2, w) * w)
    return s * yn, w * yn, s2 * yn, w2 * yn


def order_pair(S, p, q, eps):
    """p-first: GN y_A + eps p, then y_A + eps(p+q); q-first likewise. Returns thetas."""
    m, X, yA = S.model, S.X, S.yA
    t1, _ = tcr.gn_learn(m, S.theta_A, (X, yA + eps * p))
    tp, _ = tcr.gn_learn(m, t1, (X, yA + eps * (p + q)))
    t2, _ = tcr.gn_learn(m, S.theta_A, (X, yA + eps * q))
    tq, _ = tcr.gn_learn(m, t2, (X, yA + eps * (p + q)))
    return tp, tq


def curriculum(S, first, ystar, T_tot):
    """GD(1) with a total budget of T_tot steps. first=None: JOINT (all on ystar); else T/2 steps
    on `first` then T/2 on ystar (eta recomputed at the start of each learn = each phase)."""
    npm = np_model(S.model, S.X)
    C = np.zeros((N, N))
    th = S.theta_A
    infos = []
    if first is None:
        th, C, i = fo_learn_C(npm, S.model, th, S.X, ystar.numpy(), S.F_A_np, C, "GD", 1.0,
                              fixed_steps=T_tot)
        infos.append(i)
    else:
        th, C, i = fo_learn_C(npm, S.model, th, S.X, first.numpy(), S.F_A_np, C, "GD", 1.0,
                              fixed_steps=T_tot // 2)
        infos.append(i)
        th, C, i = fo_learn_C(npm, S.model, th, S.X, ystar.numpy(), S.F_A_np, C, "GD", 1.0,
                              fixed_steps=T_tot // 2)
        infos.append(i)
    return th, infos


def curriculum_lin(KA, FA, first, ystar, T_tot):
    """Frozen-kernel GD(1) in label space: F <- F - eta' K_A (F - y), eta' = 1/lambda_max(K_A)."""
    etap = 1.0 / np.linalg.eigvalsh(KA)[-1]
    F = FA.copy()
    phases = [(ystar, T_tot)] if first is None else [(first, T_tot // 2), (ystar, T_tot // 2)]
    for y, k in phases:
        for _ in range(k):
            F = F - etap * (KA @ (F - y))
    return F


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------
GAMMAS8 = (0.0, 0.5, 0.9, 0.99)
SIGMA8 = 0.05


def draw_P(seed, KA, gammas=GAMMAS8, max_draws=10000):
    """Generator(8000+seed): P = torch.rand(16,16)**4, rows normalised; redraw (sequentially from
    the same generator) until every eigenvalue of K_A M, M = I - gamma P, has real part
    > 1e-3 lambda_max(K_A) for EVERY gamma in `gammas` (one P per seed, shared by all gamma)."""
    g = torch.Generator().manual_seed(8000 + seed)
    lmax = np.linalg.eigvalsh(KA)[-1]
    for d in range(1, max_draws + 1):
        Pm = torch.rand(N, N, generator=g) ** 4
        Pm = (Pm / Pm.sum(1, keepdim=True)).numpy()
        ok = True
        minre = {}
        for gm in gammas:
            ev = np.linalg.eigvals(KA @ (np.eye(N) - gm * Pm))
            minre[gm] = float(ev.real.min())
            ok &= minre[gm] > 1e-3 * lmax
        if ok:
            return Pm, d, minre
    raise RuntimeError("no admissible P")


def lyap_quantities(KA, Pm, gamma, sigma=SIGMA8):
    lmax = np.linalg.eigvalsh(KA)[-1]
    etap = 0.2 / lmax
    M = np.eye(N) - gamma * Pm
    A = etap * KA @ M
    BB = etap ** 2 * sigma ** 2 * KA @ KA
    IA = np.eye(N) - A
    Sig = scipy.linalg.solve_discrete_lyapunov(IA, BB)
    return {"etap": etap, "M": M, "A": A, "BB": BB, "Sig": Sig, "E": -A @ Sig,
            "spec_rad_IA": float(np.abs(np.linalg.eigvals(IA)).max())}


def lyap_residual(q):
    IA = np.eye(N) - q["A"]
    Sig = q["Sig"]
    res = np.linalg.norm(Sig - IA @ Sig @ IA.T - q["BB"]) / np.linalg.norm(Sig)
    asym = np.linalg.norm(Sig - Sig.T) / np.linalg.norm(Sig)
    ev = np.linalg.eigvalsh((Sig + Sig.T) / 2)
    return float(res), float(asym), float(ev.min()), float(ev.max())


def hess_rows(S, Z, step=FD_STEP):
    """Column a of Z (P x n): H_a z_a = FD derivative of row a of J along z_a (s = 1e-4/||z_a||).
    Returns sum_a H_a z_a (P,)."""
    tot = torch.zeros(S.model.P)
    for a in range(Z.shape[1]):
        z = Z[:, a]
        nz = nrm(z)
        if nz == 0:
            continue
        s = step / nz
        Jp = S.model.jac(S.theta_A + s * z, S.X)[a]
        Jm = S.model.jac(S.theta_A - s * z, S.X)[a]
        tot += (Jp - Jm) / (2 * s)
    return tot


def drift_predictions(S, q, Pm, gamma):
    """d_law = LAW(E); d_dir; the two potential terms t1, t2 so that
    d_pot(kappa) = -(eta'/2) P_v [2 t1 - kappa gamma t2]; least-squares kappa*."""
    L = law(S)
    etap = q["etap"]
    Sig = torch.from_numpy(q["Sig"])
    M = torch.from_numpy(q["M"])
    Pt = torch.from_numpy(Pm)
    Xi = L.XI                                     # J^T K^{-1}  (P x n)
    d_law = L(q["E"])
    Z_dir = Xi @ (Sig @ M.T)                      # col a = Sigma_theta (g_a - gamma sum_b P_ab g_b)
    d_dir = -etap * L.Pv(hess_rows(S, Z_dir))
    t1 = L.Pv(hess_rows(S, Xi @ Sig))             # sum_a H_a Sigma_theta g_a
    t2 = L.Pv(hess_rows(S, Xi @ (Sig @ Pt.T)))    # sum_a H_a Sigma_theta (sum_b P_ab g_b)
    a_vec = -etap * t1
    b_vec = 0.5 * etap * gamma * t2
    out = {"d_law": d_law, "d_dir": d_dir, "a": a_vec, "b": b_vec,
           "d_pot1": a_vec + 1.0 * b_vec, "d_pot2": a_vec + 2.0 * b_vec}
    bb = torch.dot(b_vec, b_vec).item()
    out["kappa_star"] = (torch.dot(b_vec, d_law - a_vec).item() / bb) if bb > 0 else float("nan")
    Es, Ea = c_parts(q["E"])
    out["law_E_sym"] = L(Es)
    out["law_E_area"] = L(Ea)
    return out


class NpMLP6(tcr4.NpMLP):
    """Adds a vector-Jacobian product J^T e to tcr4.NpMLP (same backprop code as NpMLP.fg)."""

    def fwd(self, t):
        sd, sh = self.sd, self.sh
        W1, b1, W2, b2, w3 = (self._v(t, k) for k in ("W1", "b1", "W2", "b2", "w3"))
        a1 = np.tanh(self.Xn @ W1.T / sd + b1)
        a2 = np.tanh(a1 @ W2.T / sh + b2)
        return a2 @ w3 / sh + t[self.b3i], (a1, a2)

    def vjp(self, t, acts, e):
        sd, sh = self.sd, self.sh
        a1, a2 = acts
        W2, w3 = self._v(t, "W2"), self._v(t, "w3")
        g = np.empty(self.P)
        self._v(g, "w3")[:] = a2.T @ e / sh
        g[self.b3i] = e.sum()
        dz2 = (e[:, None] * w3[None, :] / sh) * (1.0 - a2 * a2)
        self._v(g, "W2")[:] = dz2.T @ a1 / sh
        self._v(g, "b2")[:] = dz2.sum(0)
        dz1 = (dz2 @ W2 / sh) * (1.0 - a1 * a1)
        self._v(g, "W1")[:] = dz1.T @ self.Xn / sd
        self._v(g, "b1")[:] = dz1.sum(0)
        return g


def td_noise(seed, stream, n_steps):
    """zeta_k ~ N(0, I_16), k = 0..n_steps-1, from Generator(8100 + seed + stream), drawn as one
    (n_steps, 16) tensor."""
    g = torch.Generator().manual_seed(8100 + seed + stream)
    return torch.randn(n_steps, N, generator=g).numpy()


def td_simulate(S, npm, M, etap, zeta, sigma=SIGMA8):
    """theta <- theta - eta' J^T (M (F - y_A) - sigma zeta_k) for len(zeta) steps from theta_A,
    then a clean GN return to y_A. Returns (theta_end, measured C incl. the return, info)."""
    yA = S.yA.numpy()
    FA = S.F_A_np
    th = S.theta_A.numpy().copy()
    C = np.zeros((N, N))
    F, acts = npm.fwd(th)
    blew = False
    for k in range(zeta.shape[0]):
        v = M @ (F - yA) - sigma * zeta[k]
        th -= etap * npm.vjp(th, acts, v)
        F_new, acts = npm.fwd(th)
        C += np.outer(F_new - F, F - FA)
        F = F_new
        if not np.all(np.isfinite(F)):
            blew = True
            break
    pre_res = float(np.abs(F - yA).max())
    th_t = torch.from_numpy(th.copy())
    th_end, C, info = gn_learn_C(S.model, th_t, S.X, S.yA, FA, C)
    info["pre_return_res_inf"] = pre_res
    info["blew_up"] = blew
    return th_end, C, info


def transient_expected_C(q, n_steps):
    """Exact expectation of the measured C under the linearised dynamics from x_0 = 0, including
    the final GN return step:  E[C] = -A sum_{k<N} Sigma_k - Sigma_N."""
    A, BB = q["A"], q["BB"]
    IA = np.eye(N) - A
    Sk = np.zeros((N, N))
    acc = np.zeros((N, N))
    for _ in range(n_steps):
        acc += Sk
        Sk = IA @ Sk @ IA.T + BB
    return -A @ acc - Sk


def kernel_alignment(model, theta, X, Pm):
    K = K_of(model, theta, X).numpy()
    d = np.sqrt(np.diag(K))
    return float((Pm * K / np.outer(d, d)).sum() / N), float(np.trace(K))


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------
def drive_target(yA, a, p, q, omega, k, kind):
    if kind == "ROT":
        return yA + a * (math.cos(omega * k) * p + math.sin(omega * k) * q)
    if kind == "LINE":
        return yA + a * math.cos(omega * k) * (p + q)
    raise ValueError(kind)


def run_drive(S, npm, eta, p, q, omega, Pi, kind, m_list, a=0.05, check=True):
    """GD (eta fixed) tracking the drive, one step per k = 0, 1, ...; after m full periods
    (k = m*Pi steps) a clean GN return to y_A is done from a copy. Returns
    {m: (theta_end, C_measured_incl_return, gn_info)}."""
    yA = S.yA.numpy()
    pn, qn = p.numpy(), q.numpy()
    FA = S.F_A_np
    th = S.theta_A.numpy().copy()
    C = np.zeros((N, N))
    out = {}
    M = max(m_list)
    F, r, g, _ = npm.fg(th, drive_target(yA, a, pn, qn, omega, 0, kind))
    for k in range(M * Pi):
        th = th - eta * g                     # step k uses target y_k (g = J^T (F - y_k) / n)
        F_new, r, g, _ = npm.fg(th, drive_target(yA, a, pn, qn, omega, k + 1, kind))
        C += np.outer(F_new - F, F - FA)
        F = F_new
        if (k + 1) % Pi == 0 and (k + 1) // Pi in m_list:
            m = (k + 1) // Pi
            th_t = torch.from_numpy(th.copy())
            th_end, Cm, info = gn_learn_C(S.model, th_t, S.X, S.yA, FA, C.copy())
            if check:
                tcr3.end_check(S.model, th_end, S.A)
            out[m] = (th_end, Cm, info)
    return out


def apriori_drive_C(KA, FA, yA, eta_over_n, p, q, omega, Pi, kind, m_list, a=0.05):
    """Linearised label-space recursion F <- F - (eta/n) K_A (F - y_k) with path matrix, then the
    a-priori GN return (one step e = -(F - y_A)). Returns {m: C}."""
    pn, qn = p.numpy(), q.numpy()
    F = FA.copy()
    C = np.zeros((N, N))
    out = {}
    for k in range(max(m_list) * Pi):
        y = drive_target(yA, a, pn, qn, omega, k, kind)
        e = -eta_over_n * (KA @ (F - y))
        C += np.outer(e, F - FA)
        F = F + e
        if (k + 1) % Pi == 0 and (k + 1) // Pi in m_list:
            e = -(F - yA)
            out[(k + 1) // Pi] = C + np.outer(e, F - FA)
    return out


def steered_plane(S, x1):
    """S_ij = J_x1 . T_ij; A_bar = antisymmetric part; eigenvector of A_bar with the largest
    imaginary eigenvalue; (p, q) = sqrt(2) (Re z, Im z), each normalised to ||y_A||."""
    L = law(S)
    g1 = S.model.jac(S.theta_A, x1[None, :])[0]
    Tm = T_matrix(L)
    Smat = (g1 @ Tm).reshape(N, N).numpy()              # [i, j] = J_x1 . T_ij
    Abar = (Smat - Smat.T) / 2
    ev, evec = np.linalg.eig(Abar)
    k = int(np.argmax(ev.imag))
    z = evec[:, k]
    p = np.sqrt(2) * z.real
    q = np.sqrt(2) * z.imag
    yn = torch.linalg.norm(S.yA).item()
    p = torch.from_numpy(p / np.linalg.norm(p) * yn)
    q = torch.from_numpy(q / np.linalg.norm(q) * yn)
    return p, q, {"mu_max": float(ev.imag[k]), "pq_cos": float(cos(p, q)),
                  "eig_imag_sorted": sorted(ev.imag.tolist(), reverse=True)[:4],
                  "Smat_antisym_F": float(np.linalg.norm(Abar)),
                  "Smat_sym_F": float(np.linalg.norm((Smat + Smat.T) / 2))}
