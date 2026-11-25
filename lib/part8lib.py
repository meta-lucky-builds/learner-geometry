"""part8lib - library of alignment_fraction_test.ipynb (SPEC8: test of the alignment-fraction law).
Written by the notebook's %%writefile cells; imported by the notebook kernel and by every worker process.
Edit the notebook cells, not this file. Every building block takes callables F(theta), J(theta) (a Problem),
so the SAME code runs on the solvable 2-parameter model, the small tanh network, the linear control and the MLP."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (must precede CUDA init)
import sys, re, json, time, math, hashlib, platform, traceback, subprocess, glob, gc, zipfile, shutil, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.func import jacrev

torch.set_default_dtype(torch.float64)            # float64 EVERYWHERE (spec sec. 2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
DT = torch.float64

try:
    import torchvision
    HAVE_TORCHVISION = True
except Exception:
    HAVE_TORCHVISION = False
try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

CFG = {}
DEVICE = torch.device("cpu")
DATA_SOURCE = None


def configure(cfg):
    """Install the CONFIG-cell settings as module globals (kernel: from CFG; workers: from config.json)."""
    CFG.clear()
    CFG.update(cfg)
    globals().update(cfg)


def set_device(dev):
    global DEVICE
    DEVICE = torch.device(dev)


def gen(seed):
    """CPU torch.Generator with a fixed seed (every spec seed goes through this)."""
    return torch.Generator().manual_seed(int(seed))


def say(*a):
    print(*a, flush=True)


def fmt_time(s):
    s = float(s)
    if s < 120:
        return f"{s:.1f}s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def free_memory():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


_HB = {"t": time.time(), "label": ""}


def heartbeat(msg="", force=False):
    """Progress line at least every PROGRESS_EVERY_S seconds (called from inside long loops)."""
    now = time.time()
    if force or now - _HB["t"] >= CFG.get("PROGRESS_EVERY_S", 60):
        _HB["t"] = now
        say(f"  [{time.strftime('%H:%M:%S')}] {_HB['label']} {msg}")


def test_fail_hook(name):
    """TEST HOOK, inert unless the environment variable PART8_TEST_FAIL_KEY is set: a case whose name contains
    that string raises on purpose (used only to check the failure listing)."""
    key = os.environ.get("PART8_TEST_FAIL_KEY", "")
    if key and key in name:
        raise RuntimeError(f"deliberate failure injected by PART8_TEST_FAIL_KEY={key!r} (test hook)")


# =============================================================================================================
# Generic problem: F(theta) (N,), J(theta) (N,P); optional fast forward/backward for first-order learners
# =============================================================================================================
class Problem:
    def __init__(self, F, J, N, P, C=1, fwd=None, bwd=None, name=""):
        self._F, self._J, self._fwd, self._bwd = F, J, fwd, bwd
        self.N, self.P, self.C, self.name = int(N), int(P), int(C), name
        self.counts = {"F": 0, "J": 0, "VJP": 0}

    def F(self, th):
        self.counts["F"] += 1
        return self._F(th)

    def J(self, th):
        self.counts["J"] += 1
        return self._J(th)

    def fwd(self, th):
        """(F(theta), cache) - cache feeds vjp()."""
        self.counts["F"] += 1
        if self._fwd is None:
            return self._F(th), th
        return self._fwd(th)

    def vjp(self, cache, r):
        """J(theta)^T r for the theta of the cache."""
        self.counts["VJP"] += 1
        if self._bwd is None:
            return self._J(cache).T @ r
        return self._bwd(cache, r)


def _act(name, z):
    if name == "tanh":
        return torch.tanh(z)
    if name == "gelu":
        return Fnn.gelu(z)                      # exact (erf) GELU, PyTorch default
    if name == "relu":
        return torch.relu(z)
    raise ValueError(name)


def _act_deriv(name, z, h):
    if name == "tanh":
        return 1.0 - h * h
    if name == "gelu":
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))) + z * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    if name == "relu":
        return (z > 0).to(z.dtype)
    raise ValueError(name)


def mlp_forward(th, X, shapes, act, keep=False):
    """Layered network on a FLAT parameter vector (order = nn.Linear weight, bias per layer, as named_parameters).
    Returns outputs flattened row-major: index a = (input index) * C + (class index)."""
    o, h, cache = 0, X, []
    nl = len(shapes)
    for l, (wo, wi) in enumerate(shapes):
        W = th[o:o + wo * wi].view(wo, wi)
        o += wo * wi
        b = th[o:o + wo]
        o += wo
        z = h @ W.T + b
        if l < nl - 1:
            hn = _act(act, z)
            if keep:
                cache.append((h, W, z, hn))
            h = hn
        else:
            if keep:
                cache.append((h, W, None, None))
            h = z
    return h.reshape(-1), cache


def mlp_backward(cache, r, C, act):
    """J^T r by explicit back-propagation (checked against jacrev in TESTS U1b)."""
    g = r.view(-1, C)
    grads = []
    for l in range(len(cache) - 1, -1, -1):
        h_in, W, _, _ = cache[l]
        grads.append(g.sum(0))
        grads.append((g.T @ h_in).reshape(-1))
        if l > 0:
            _, _, z_prev, h_prev = cache[l - 1]
            g = (g @ W) * _act_deriv(act, z_prev, h_prev)
    grads.reverse()                               # appended b_L, W_L, ..., b_0, W_0 -> reversed: W_0, b_0, ..., W_L, b_L
    return torch.cat(grads)


def layered_problem(X, sizes, act, seed, name=""):
    """MLP sizes[0] -> ... -> sizes[-1] with biases, default nn.Linear init under torch.manual_seed(seed).
    Returns (Problem, theta_init)."""
    torch.manual_seed(int(seed))
    layers = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]   # same RNG order as nn.Sequential
    shapes, parts = [], []
    for L in layers:
        shapes.append((L.out_features, L.in_features))
        parts += [L.weight.detach().reshape(-1), L.bias.detach().reshape(-1)]
    th = torch.cat(parts).to(DT).to(X.device)
    C = sizes[-1]
    N = X.shape[0] * C
    chunk = CFG.get("JAC_CHUNK", None)

    def F(t):
        return mlp_forward(t, X, shapes, act)[0]

    jf = jacrev(F, chunk_size=chunk) if chunk else jacrev(F)

    def J(t):
        return jf(t)

    def fwd(t):
        out, cache = mlp_forward(t, X, shapes, act, keep=True)
        return out, cache

    def bwd(cache, r):
        return mlp_backward(cache, r, C, act)

    prob = Problem(F, J, N, th.numel(), C=C, fwd=fwd, bwd=bwd, name=name)
    prob.shapes, prob.act, prob.X = shapes, act, X
    return prob, th


def toy_problem():
    """Exactly solvable model F(u, v) = u^2 - v^2 (N = 1, P = 2), theta0 = (1, 1)."""
    def F(t):
        return (t[0] ** 2 - t[1] ** 2).reshape(1)

    def J(t):
        return torch.stack([2 * t[0], -2 * t[1]]).reshape(1, 2)

    return Problem(F, J, 1, 2, C=1, name="u2-v2"), torch.tensor([1.0, 1.0], dtype=DT, device=DEVICE)


# =============================================================================================================
# Linear algebra
# =============================================================================================================
def gram(J):
    K = J @ J.T
    return 0.5 * (K + K.T)


def chol(K):
    return torch.linalg.cholesky(K)


def csolve(L, r):
    if r.dim() == 1:
        return torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
    return torch.cholesky_solve(r, L)


def eig_desc(K):
    lam, Q = torch.linalg.eigh(K)
    return lam.flip(0), Q.flip(1)


def vnorm(x):
    return float(torch.linalg.vector_norm(x))


class Start:
    """theta0 with J0, K0, its Cholesky factor and eigen-decomposition (computed ONCE per start)."""

    def __init__(self, prob, th0, tag=""):
        self.prob, self.th0, self.tag = prob, th0.clone(), tag
        self.F0 = prob.F(self.th0)
        self.J0 = prob.J(self.th0)
        self.K0 = gram(self.J0)
        self.lam, self.Q = eig_desc(self.K0)
        self.lam1 = float(self.lam[0])
        self.lamN = float(self.lam[-1])
        self.trK = float(torch.trace(self.K0))
        self.ill = (self.lamN / self.lam1) < CFG.get("ILL_COND", 1e-11)
        self.L0 = None if self.ill else chol(self.K0)
        self.N, self.P = prob.N, prob.P

    def V0(self, w):
        """Vertical projector AT THE START: V0 w = w - J0^T K0^{-1} J0 w (spec 3.1)."""
        return w - self.J0.T @ csolve(self.L0, self.J0 @ w)

    def spectrum(self):
        return dict(lam1=self.lam1, lamN=self.lamN, trK=self.trK, cond=self.lam1 / self.lamN,
                    lamN_over_lam1=self.lamN / self.lam1, ill_conditioned=bool(self.ill))

# =============================================================================================================
# 3.2 Exact jump
# =============================================================================================================
def jump(prob, thb, t, Jb=None):
    """theta = theta_b + J_b^T a with F(theta) = t. Chord iteration a <- a + K_b^{-1}(t - F) (K_b factorised once),
    stop at ||F - t||_inf <= JUMP_TOL; if not reached in JUMP_CHORD_CAP iterations, Newton on a with
    M = J(theta) J_b^T (recomputed each iteration, cap JUMP_NEWTON_CAP)."""
    if Jb is None:
        Jb = prob.J(thb)
    Lb = chol(gram(Jb))
    tol = CFG["JUMP_TOL"]
    a = torch.zeros(prob.N, dtype=DT, device=thb.device)
    th = thb.clone()
    r = t - prob.F(th)
    rinf = float(r.abs().max())
    best = (rinf, a.clone())
    it_chord, it_newton, status = 0, 0, None
    if rinf <= tol:
        status = "CONVERGED"
    r0 = rinf
    while status is None and it_chord < CFG["JUMP_CHORD_CAP"]:
        a = a + csolve(Lb, r)
        it_chord += 1
        th = thb + Jb.T @ a
        r = t - prob.F(th)
        rinf = float(r.abs().max())
        if math.isfinite(rinf) and rinf < best[0]:
            best = (rinf, a.clone())
        if rinf <= tol:
            status = "CONVERGED"
        elif not math.isfinite(rinf) or rinf > 1e6 * max(r0, 1e-300):
            break                                # chord iteration diverged: go to Newton from the best a seen
    if status is None:
        a = best[1].clone()
        th = thb + Jb.T @ a
        r = t - prob.F(th)
        while it_newton < CFG["JUMP_NEWTON_CAP"]:
            M = prob.J(th) @ Jb.T
            try:
                a = a + torch.linalg.solve(M, r)
            except RuntimeError:                         # singular Newton matrix -> FAILED (recorded, never hidden)
                break
            it_newton += 1
            th = thb + Jb.T @ a
            r = t - prob.F(th)
            rinf = float(r.abs().max())
            if rinf <= tol:
                status = "CONVERGED"
                break
            if not math.isfinite(rinf):
                break
        if status is None:
            status = "FAILED"
    return dict(theta=th, status=status, it_chord=it_chord, it_newton=it_newton, res_inf=rinf)


# =============================================================================================================
# 3.3 Learners
# =============================================================================================================
def learner_name(L):
    k = L["kind"]
    if k == "SL":
        return f"SL({L['m']})"
    if k == "DJ":
        return f"DJ({L['c']:g})"
    if k == "GD":
        return f"GD({L['c']:g})"
    if k == "HB":
        return f"HB({L['c']:g},{L['mu']:g})"
    if k == "SGD":
        return f"SGD({L['c']:g},{L['B']})"
    if k == "ADAM":
        return f"ADAM({L['lr']:g})"
    if k == "PROX":
        return f"PROX({L['gt_rel']:g})"
    raise ValueError(k)


def predict_f(L, lam_ratio=None):
    """Closed-form alignment fraction (spec sec. 1 / 5); None where no closed form is claimed."""
    k = L["kind"]
    if k == "SL":
        return 0.5 * (1.0 - 1.0 / L["m"])
    if k == "DJ":
        c = L["c"]
        return (1.0 - c) / (2.0 - c)
    if k == "GD" and lam_ratio is not None:
        ci = L["c"] * lam_ratio
        return (1.0 - ci) / (2.0 - ci)
    if k == "HB" and lam_ratio is not None:
        ci, mu = L["c"] * lam_ratio, L["mu"]
        return 1.0 - ((1 - mu) ** 2 * (1 + mu) + 2 * mu * ci) / ((1 - mu) ** 2 * (2 * (1 + mu) - ci))
    if k == "PROX":
        return 1.0
    return None


def run_SL(prob, st, t, m):
    th = st.th0.clone()
    status, itc, itn, nfail = "CONVERGED", 0, 0, 0
    for k in range(1, m + 1):
        tk = st.F0 + (k / m) * (t - st.F0)
        jr = jump(prob, th, tk)
        th = jr["theta"]
        itc += jr["it_chord"]
        itn += jr["it_newton"]
        if jr["status"] != "CONVERGED":
            nfail += 1
            status = "JUMP-FAILED"
    return dict(theta=th, status=status, steps=m, jump_it_chord=itc, jump_it_newton=itn, jumps_failed=nfail)


def run_DJ(prob, st, t, c):
    th = st.th0.clone()
    target = CFG["DJ_TOL_REL"] * vnorm(t - st.F0)
    status, k = None, 0
    while True:
        r = prob.F(th) - t
        rn = vnorm(r)
        if rn <= target:
            status = "CONVERGED"
            break
        if not math.isfinite(rn):
            status = "DIVERGED"
            break
        if k >= CFG["DJ_CAP"]:
            status = "UNCONVERGED"
            break
        J = prob.J(th)
        th = th - c * (J.T @ csolve(chol(gram(J)), r))
        k += 1
    return dict(theta=th, status=status, steps=k, final_res=rn)


def run_first_order(prob, st, t, s, L):
    """GD / HB / SGD / ADAM. eta = c * N / lambda_1(K0) (fixed). Stop: ||F - t||_2 <= stop_rel * s
    (and, if u_tol_rel is set, ||u||_2 <= u_tol_rel * ||theta - theta0||), or cap steps.
    path=True accumulates C += outer(F_{k+1} - F_k, F_k - F0) and Q += outer(dF, dF) with F on the FULL set."""
    kind = L["kind"]
    N, C = prob.N, prob.C
    th = st.th0.clone()
    u = torch.zeros_like(th)
    mu = L.get("mu", 0.0)
    eta = L["c"] * N / st.lam1 if kind != "ADAM" else None
    stop_abs = L["stop_rel"] * s
    u_tol = L.get("u_tol_rel")
    cap = L["cap"]
    dir_rel = L.get("dir_stop_rel")                         # A2.1: "direction-converged" stop (None = old rule)
    if dir_rel is not None:
        dvec = (t - st.F0) / vnorm(t - st.F0)                # unit target direction d = (t - F0)/||t - F0||
        dir_abs, dir_win, dir_run = dir_rel * s, int(L["dir_window"]), 0
    path = L.get("path", False)
    Cm = torch.zeros(N, N, dtype=DT, device=th.device) if path else None
    Qm = torch.zeros(N, N, dtype=DT, device=th.device) if path else None
    if kind == "SGD":
        g = gen(7000 + L["seed"])
        n_in, B = N // C, L["B"]
        batches = []
    if kind == "ADAM":
        p = torch.nn.Parameter(th.clone())
        opt = torch.optim.Adam([p], lr=L["lr"])
    F, cache = prob.fwd(th)
    k, status = 0, None
    t0 = time.time()
    while True:
        r = F - t
        rn = vnorm(r)
        if not math.isfinite(rn) or rn > 1e8 * max(s, 1e-300):
            status = "DIVERGED"
            break
        if dir_rel is not None:                             # A2.1: |d.(F_j - t)| <= DIR_STOP_REL s for DIR_WINDOW
            dir_run = dir_run + 1 if abs(float(dvec @ r)) <= dir_abs else 0     # consecutive steps; no velocity rule
            conv = dir_run >= dir_win
        else:
            conv = rn <= stop_abs
            if conv and u_tol is not None:
                conv = vnorm(u) <= u_tol * vnorm(th - st.th0)
        if conv:
            status = "CONVERGED"
            break
        if k >= cap:
            status = "UNCONVERGED"
            break
        if kind in ("GD", "HB"):
            grad = prob.vjp(cache, r) / N
            u = mu * u - eta * grad
            th = th + u
        elif kind == "SGD":
            if not batches:
                perm = torch.randperm(n_in, generator=g)
                batches = [perm[j:j + B] for j in range(0, n_in, B)]
            idx = batches.pop(0)
            mask = torch.zeros(n_in, C, dtype=DT)
            mask[idx] = 1.0
            mask = mask.reshape(-1).to(th.device)
            grad = prob.vjp(cache, r * mask) / (len(idx) * C)
            th = th - eta * grad
        elif kind == "ADAM":
            p.grad = prob.vjp(cache, r) / N
            opt.step()
            th = p.detach().clone()
        k += 1
        Fn, cache = prob.fwd(th)
        if path:
            dF = Fn - F
            Cm += torch.outer(dF, F - st.F0)
            Qm += torch.outer(dF, dF)
        F = Fn
        if k % 2000 == 0:
            heartbeat(f"{learner_name(L)} step {k} |F-t|={rn:.3e} (target {stop_abs:.3e}) {time.time() - t0:.0f}s")
    out = dict(theta=th, status=status, steps=k, final_res=rn, u_norm=vnorm(u))
    if dir_rel is not None:
        out["final_res_over_s"] = rn / s                     # A2.1: full residual / s and ||u|| at the stop
        out["dir_res_over_s"] = abs(float(dvec @ r)) / s
    if path:
        dY = F - st.F0
        lhs = Cm + Cm.T
        rhs = torch.outer(dY, dY) - Qm
        err = vnorm(lhs - rhs) / max(vnorm(rhs), 1e-300)
        out.update(C=Cm, Q=Qm, path_identity_err=err)
        assert err <= CFG["PATH_ASSERT_REL"], f"path bookkeeping identity violated: rel err {err:.3e}"
        if kind == "GD":                                    # linearised run with the same number of steps (H5b)
            r0 = st.F0 - t
            rk = r0.clone()
            Cl = torch.zeros(N, N, dtype=DT, device=th.device)
            fac = L["c"] / st.lam1                          # eta / N
            for _ in range(k):
                rn_ = rk - fac * (st.K0 @ rk)
                Cl += torch.outer(rn_ - rk, rk - r0)
                rk = rn_
            out["C_lin"] = Cl
    return out


def run_PROX(prob, st, t, gt):
    """Stationary point of L(theta; t) + (gamma/2)||theta - theta0||^2, gamma = gt / N."""
    N = prob.N
    gam = gt / N
    g0 = vnorm(st.J0.T @ (st.F0 - t) / N)
    th = st.th0.clone()
    k, status = 0, None
    I = torch.eye(N, dtype=DT, device=th.device)
    while True:
        F = prob.F(th)
        J = prob.J(th)
        gn = vnorm(J.T @ (F - t) / N + gam * (th - st.th0))
        if gn <= CFG["PROX_TOL_REL"] * g0:
            status = "CONVERGED"
            break
        if not math.isfinite(gn):
            status = "DIVERGED"
            break
        if k >= CFG["PROX_CAP"]:
            status = "UNCONVERGED"
            break
        Kt = gram(J) + gt * I
        th = st.th0 + J.T @ csolve(chol(Kt), (t - F) + J @ (th - st.th0))
        k += 1
    return dict(theta=th, status=status, steps=k, grad_norm=gn)


def run_learner(prob, st, t, s, L):
    k = L["kind"]
    if k == "SL":
        return run_SL(prob, st, t, L["m"])
    if k == "DJ":
        return run_DJ(prob, st, t, L["c"])
    if k in ("GD", "HB", "SGD", "ADAM"):
        return run_first_order(prob, st, t, s, L)
    if k == "PROX":
        return run_PROX(prob, st, t, L["gt_rel"] * st.lam1)
    raise ValueError(k)


# =============================================================================================================
# 3.4 Nearest point on the fibre of F_L
# =============================================================================================================
def _np_eval(prob, th0, FL, th):
    F = prob.F(th)
    J = prob.J(th)
    K = gram(J)
    L = chol(K)
    w = th - th0
    Jw = J @ w
    vw = w - J.T @ csolve(L, Jw)                           # V_theta (theta - theta0): projector AT THE CURRENT theta
    th_iter = th0 + J.T @ csolve(L, (FL - F) + Jw)          # spec 3.4 iteration
    wn = vnorm(w)
    return dict(th=th, F=F, K=K, res=float((F - FL).abs().max()), stat=(vnorm(vw) / wn) if wn > 0 else 0.0,
                th_iter=th_iter, merit=vnorm(th_iter - th))


def nearest_point(prob, th0, FL, th_start, cap=None):
    """Minimise ||theta - theta0||^2 s.t. F(theta) = FL, iterating from th_start (theta_L, or theta0 for TH-type).
    Step halving (s = 1, 1/2, ..., >= NP_MIN_STEP) when the iteration's residual increases. NP_MERIT selects the
    residual: "step" = ||theta_iter - theta|| (the residual of the fixed-point iteration; zero exactly at the
    nearest point), "output" = ||F - FL||_inf (literal reading; see the notebook's note on SPEC ERROR?)."""
    cap = CFG["NP_CAP"] if cap is None else int(cap)        # A1.6: R2 cases pass NP_CAP_R2 explicitly
    cur = _np_eval(prob, th0, FL, th_start)
    K_start = cur["K"]
    it, halv, status = 0, 0, None
    key = "merit" if CFG.get("NP_MERIT", "step") == "step" else "res"
    while True:
        if cur["res"] <= CFG["NP_RES_TOL"] and cur["stat"] <= CFG["NP_STAT_TOL"]:
            status = "CONVERGED"
            break
        if it >= cap or not math.isfinite(cur["res"]):
            status = "NP-FAILED"
            break
        s = 1.0
        while True:
            th_new = cur["th_iter"] if s == 1.0 else cur["th"] + s * (cur["th_iter"] - cur["th"])
            cand = _np_eval(prob, th0, FL, th_new)
            if not (cand[key] > cur[key]) or s <= CFG["NP_MIN_STEP"]:
                break
            s *= 0.5
            halv += 1
        cur = cand
        it += 1
    return dict(theta=cur["th"], status=status, it=it, halvings=halv, stat=cur["stat"], res=cur["res"],
                K_end=cur["K"], K_start=K_start)


# =============================================================================================================
# 3.5 The measurement (same function for every learner and every problem)
# =============================================================================================================
def _lam1_tr(K):
    return float(torch.linalg.eigvalsh(K)[-1]), float(torch.trace(K))


def measure(prob, st, thL, np_cap=None):
    """Returns (scalars, v_L, v_NP). theta_NP is the nearest point of the learner's OWN final outputs F_L."""
    FL = prob.F(thL)
    dY = FL - st.F0
    npr = nearest_point(prob, st.th0, FL, thL, cap=np_cap)
    thNP = npr["theta"]
    thr = jump(prob, st.th0, FL, Jb=st.J0)                  # TH = JUMP(theta0, F_L)
    thTH = thr["theta"]
    vL = st.V0(thL - st.th0)
    vNP = st.V0(thNP - st.th0)
    nL, nNP = vnorm(vL), vnorm(vNP)
    dot = float(vL @ vNP)
    rho = dot / (nNP * nNP) if nNP > 0 else float("nan")
    cos = dot / (nL * nNP) if nL > 0 and nNP > 0 else float("nan")
    dL, dNP, dTH = vnorm(thL - st.th0), vnorm(thNP - st.th0), vnorm(thTH - st.th0)
    floor = vnorm(st.V0(thTH - st.th0))
    below = nNP < CFG["FLOOR_FACTOR"] * max(floor, CFG["FLOOR_REL"] * dNP)
    invalid = dNP > dL * (1.0 + CFG["NP_DIST_SLACK"])
    lL, tL = _lam1_tr(npr["K_start"])                       # NP starts at theta_L -> K(theta_L)
    lNP, tNP = _lam1_tr(npr["K_end"])
    lTH, tTH = _lam1_tr(gram(prob.J(thTH)))
    l0, t0 = _lam1_tr(st.K0)

    def R(gL, gNP, gTH):
        den = gNP - gTH
        return (gL - gTH) / den if den != 0 else float("nan")

    sc = dict(rho=rho, cos=cos, nvL=nL, nvNP=nNP, dL=dL, dNP=dNP, dTH=dTH, dY=vnorm(dY),
              np_it=npr["it"], np_stat=npr["stat"], np_res=npr["res"], np_status=npr["status"],
              np_halvings=npr["halvings"], th_status=thr["status"], th_it=thr["it_chord"] + thr["it_newton"],
              floor=floor, below_resolution=bool(below), invalid=bool(invalid),
              lam1_L=lL, lam1_NP=lNP, lam1_TH=lTH, lam1_0=l0, trK_L=tL, trK_NP=tNP, trK_TH=tTH, trK_0=t0,
              R_lam1=R(lL, lNP, lTH), R_trK=R(tL, tNP, tTH), vL_over_dL=nL / dL if dL > 0 else float("nan"),
              vNP_over_dNP=nNP / dNP if dNP > 0 else float("nan"))
    return sc, vL, vNP


def even_stats(vLp, vNPp, vLm, vNPm):
    """EVEN PART (spec 3.5): v^e = (v(+) + v(-))/2 separately for v_L and v_NP, each run with its own F_L and NP."""
    vLe = 0.5 * (vLp + vLm)
    vNe = 0.5 * (vNPp + vNPm)
    a, b = vnorm(vLe), vnorm(vNe)
    dot = float(vLe @ vNe)
    return dict(rho=dot / (b * b) if b > 0 else float("nan"),
                cos=dot / (a * b) if a > 0 and b > 0 else float("nan"),
                nr=a / b if b > 0 else float("nan"), nvLe=a, nvNPe=b)


def usable(sc, need_converged=False):
    """Why a single run is excluded from rules ('' = usable)."""
    st = sc.get("learner_status")
    if st in ("DIVERGED", "JUMP-FAILED", "FAILED"):
        return st
    if need_converged and st != "CONVERGED":
        return "UNCONVERGED" if st == "UNCONVERGED" else str(st)
    if sc.get("np_status") != "CONVERGED":
        return "NP-FAILED"
    if sc.get("th_status") != "CONVERGED":
        return "TH-JUMP-FAILED"                 # the resolution floor and R_g need theta_TH
    if sc.get("invalid"):
        return "INVALID"
    if sc.get("below_resolution"):
        return "BELOW-RESOLUTION"
    return ""


def run_and_measure(prob, st, t, s, L, np_cap=None):
    """One learner run toward the nominal target t (size s), then the measurement. Returns (scalars, vectors)."""
    c0 = dict(prob.counts)
    t1 = time.time()
    res = run_learner(prob, st, t, s, L)
    t2 = time.time()
    if res["status"] == "DIVERGED" or not bool(torch.isfinite(res["theta"]).all()):
        sc = dict(learner_status="DIVERGED", steps=res.get("steps"), np_status="SKIPPED")
        vecs = {}
    else:
        sc, vL, vNP = measure(prob, st, res["theta"], np_cap=np_cap)
        sc["learner_status"] = res["status"]
        vecs = dict(vL=vL, vNP=vNP)
    t3 = time.time()
    for k, v in res.items():
        if k in ("theta", "status"):
            continue
        if torch.is_tensor(v):
            vecs[k] = v
        else:
            sc["learner_" + k if k not in ("steps",) else k] = v
    sc.update(t_learner=t2 - t1, t_measure=t3 - t2,
              n_J=prob.counts["J"] - c0["J"], n_F=prob.counts["F"] - c0["F"], n_VJP=prob.counts["VJP"] - c0["VJP"])
    return sc, vecs


def run_pair(prob, st, d, s, L):
    """Both signs t = F0 +- s d; returns (even stats, [sc+, sc-], [vecs+, vecs-])."""
    out_sc, out_v = [], []
    for sg in (1, -1):
        sc, v = run_and_measure(prob, st, st.F0 + sg * s * d, s, L)
        out_sc.append(sc)
        out_v.append(v)
    if "vL" in out_v[0] and "vL" in out_v[1]:
        ev = even_stats(out_v[0]["vL"], out_v[0]["vNP"], out_v[1]["vL"], out_v[1]["vNP"])
    else:
        ev = dict(rho=float("nan"), cos=float("nan"), nr=float("nan"))
    return ev, out_sc, out_v


# =============================================================================================================
# 3.6 Path matrix law
# =============================================================================================================
def xi(prob, th, A):
    """xi_e(theta) = J(theta)^T K(theta)^{-1} e for the columns e of A (or a single vector)."""
    J = prob.J(th)
    return J.T @ csolve(chol(gram(J)), A)


def Dxi(prob, st, a, w):
    """Directional derivative D xi_a[w] by central differences of length LAW_DELTA along w/||w||, times ||w||."""
    d = CFG["LAW_DELTA"]
    wn = vnorm(w)
    if wn == 0:
        return torch.zeros(prob.P, dtype=DT, device=w.device)
    wh = w / wn
    xp = xi(prob, st.th0 + d * wh, a)
    xm = xi(prob, st.th0 - d * wh, a)
    return (xp - xm) * (wn / (2 * d))


def law(prob, st, Cmat):
    """LAW(C) = V0 sum_k S_k D xi_{a_k}[ xi_{b_k}(theta0) ], C = sum_k S_k a_k b_k^T (SVD);
    a_k = LEFT singular vector = INCREMENT side, b_k = RIGHT singular vector = POSITION side."""
    U, S, Vh = torch.linalg.svd(Cmat)
    if float(S[0]) == 0:
        return torch.zeros(prob.P, dtype=DT, device=Cmat.device), dict(kept=0, kept_frac=1.0)
    keep = [k for k in range(S.numel()) if float(S[k]) >= CFG["LAW_SV_REL"] * float(S[0])][:CFG["LAW_MAX_TERMS"]]
    tot = torch.zeros(prob.P, dtype=DT, device=Cmat.device)
    for k in keep:
        a = U[:, k]
        b = Vh[k, :]
        w = st.J0.T @ csolve(st.L0, b)                      # xi_b(theta0)
        tot = tot + float(S[k]) * Dxi(prob, st, a, w)
        heartbeat(f"LAW term {k + 1}/{len(keep)}")
    frac = float((S[keep] ** 2).sum() / (S ** 2).sum())
    return st.V0(tot), dict(kept=len(keep), kept_frac=frac)


def law_columns(prob, st, Cmat):
    """Direct column-by-column evaluation sum_j V0 D xi_{C e_j}[ xi_{e_j} ] (unit test U13)."""
    tot = torch.zeros(prob.P, dtype=DT, device=Cmat.device)
    for j in range(Cmat.shape[1]):
        e = torch.zeros(prob.N, dtype=DT, device=Cmat.device)
        e[j] = 1.0
        w = st.J0.T @ csolve(st.L0, e)
        tot = tot + Dxi(prob, st, Cmat[:, j], w)
    return st.V0(tot)


def cosine(a, b):
    na, nb = vnorm(a), vnorm(b)
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else float("nan")

# =============================================================================================================
# Data, starts, targets
# =============================================================================================================
_MNIST_RAW = None


def _load_mnist_raw():
    global _MNIST_RAW, DATA_SOURCE
    if _MNIST_RAW is not None:
        return _MNIST_RAW
    force = CFG.get("FORCE_DATA_SOURCE")          # workers use what the kernel used
    if HAVE_TORCHVISION and force != "synthetic":
        try:
            ds = torchvision.datasets.MNIST(root=CFG.get("MNIST_ROOT", "./mnist_data"), train=True,
                                            download=bool(CFG.get("MNIST_DOWNLOAD", True)) and force is None)
            _MNIST_RAW = (ds.data.clone(), ds.targets.clone())
            DATA_SOURCE = "mnist"
        except Exception as e:
            say("MNIST could not be loaded:", repr(e))
    if _MNIST_RAW is None:
        if force == "mnist":
            raise RuntimeError("worker could not read the MNIST files the notebook kernel loaded")
        _MNIST_RAW = "synthetic"
        DATA_SOURCE = "synthetic"
        say("\n" + "!" * 96 + "\n!!! WARNING: MNIST COULD NOT BE LOADED (no torchvision or no network). FALLING BACK TO SYNTHETIC\n"
            "!!! N(0,1) INPUTS WITH RANDOM LABELS. Every result is stored with data_source='synthetic'.\n"
            "!!! (The law does not depend on the data; synthetic results are valid but are NOT MNIST results.)\n"
            + "!" * 96 + "\n")
    return _MNIST_RAW


_DATA = {}


def get_data(seed, n):
    """X (n,196) standardised with ONE scalar mean and ONE scalar std of the selected images, labels, Y (n*C,)."""
    key = (int(seed), int(n))
    if key in _DATA:
        return _DATA[key]
    raw = _load_mnist_raw()
    g = gen(1000 + seed)
    C = CFG["N_CLASSES"]
    if isinstance(raw, str):
        x = torch.randn(n, 196, generator=g, dtype=DT)
        labels = torch.randint(0, C, (n,), generator=g)
    else:
        data, targets = raw
        perm = torch.randperm(data.shape[0], generator=g)
        idx = perm[:n]
        img = data[idx].to(DT).div(255.0).unsqueeze(1)
        x = Fnn.avg_pool2d(img, 2).reshape(n, 196)           # 28x28 -> 14x14 (2x2 mean)
        labels = targets[idx].long()
    x = (x - x.mean()) / x.std(unbiased=False)
    Y = Fnn.one_hot(labels, C).to(DT).reshape(-1)
    out = (x.to(DEVICE), labels, Y.to(DEVICE))
    _DATA[key] = out
    return out


def combo_name(cb):
    return f"{cb['act']}-{cb['start']}-s{cb['seed']}"


def combo_problem(cb):
    X, labels, Y = get_data(cb["seed"], CFG["N_TRAIN"])
    if cb["act"] == "linear":
        sizes = [196, CFG["N_CLASSES"]]
        act = "tanh"                                             # unused (no hidden layer)
    else:
        sizes = [196, CFG["WIDTH"], CFG["WIDTH"], CFG["N_CLASSES"]]
        act = cb["act"]
    prob, th = layered_problem(X, sizes, act, cb["seed"], name=combo_name(cb))
    return prob, th, Y, labels


def fit_dj(prob, th, Y, c=None, mse_tol=None, cap=None):
    """DJ(c) iterations toward the targets Y until mse = ||F - Y||^2 / N <= mse_tol (cap)."""
    c = CFG["FIT_C"] if c is None else c
    mse_tol = CFG["FIT_MSE"] if mse_tol is None else mse_tol
    cap = CFG["FIT_CAP"] if cap is None else cap
    it = 0
    while True:
        r = prob.F(th) - Y
        mse = float(r @ r) / prob.N
        if mse <= mse_tol or it >= cap or not math.isfinite(mse):
            break
        J = prob.J(th)
        th = th - c * (J.T @ csolve(chol(gram(J)), r))
        it += 1
    return th, dict(fit_iterations=it, fit_mse=mse, fit_cap_hit=bool(mse > mse_tol))


def make_directions(st, seed, n):
    """Unit target directions in R^N (spec sec. 4) and S0."""
    C = st.prob.C
    assert st.N == n * C
    lab = torch.randint(0, C, (n,), generator=gen(2000 + seed))
    Ynew = Fnn.one_hot(lab, C).to(DT).reshape(-1).to(st.F0.device)
    S0 = vnorm(Ynew - st.F0)
    dirs, info = {}, {}
    ratios = (st.lam / st.lam1).cpu()
    used = set()
    for nm, target in (("eigA", 1.0), ("eigB", 0.3), ("eigC", 0.1), ("eigD", 0.01)):
        i = int(torch.argmin((ratios - target).abs()))
        while i in used and i + 1 < st.N:
            i += 1                                               # next unused index
        used.add(i)
        dirs[nm] = st.Q[:, i].clone()
        info[nm] = dict(index=i + 1, lam_ratio=float(ratios[i]))  # index reported 1-based (q_1 = top)
    dirs["label"] = (Ynew - st.F0) / S0
    info["label"] = dict(index=None, lam_ratio=None)
    rv = torch.randn(st.N, generator=gen(3000 + seed), dtype=DT).to(st.F0.device)
    dirs["rand"] = rv / vnorm(rv)
    info["rand"] = dict(index=None, lam_ratio=None)
    return dirs, info, S0


# =============================================================================================================
# Files
# =============================================================================================================
SCIENCE_KEYS = ["MODE", "N_TRAIN", "WIDTH", "N_CLASSES", "ACTS", "STARTS", "SEEDS", "JUMP_TOL", "JUMP_CHORD_CAP",
                "JUMP_NEWTON_CAP", "DJ_TOL_REL", "DJ_CAP", "NP_RES_TOL", "NP_STAT_TOL", "NP_CAP", "NP_MIN_STEP",
                "NP_MERIT", "NP_DIST_SLACK", "PROX_TOL_REL", "PROX_CAP", "FIT_C", "FIT_MSE", "FIT_CAP", "ILL_COND",
                "FLOOR_FACTOR", "FLOOR_REL", "LAW_DELTA", "LAW_SV_REL", "LAW_MAX_TERMS", "PATH_ASSERT_REL", "S1_FRAC",
                "S2_FRAC", "SL_MS", "DJ_CS", "GD_CS", "GD_STOP_REL", "GD_CAP", "HB_CS", "HB_MU", "HB_U_TOL_REL",
                "H5_STOP_REL", "H5_CAP", "SGD_C", "SGD_B", "ADAM_LR", "R1_STOP_REL", "R1_CAP", "PROX_GT_REL",
                "R2_SIZES", "R2_STOP_REL", "R2_CAP", "LIN_GD_C", "LIN_HB_C",
                "NP_CAP_R2", "ORDER_RATIO_RANGE", "ORDER_MIN_ERR", "AMENDMENT", "DIR_STOP_REL", "DIR_WINDOW"]


def config_signature():
    d = {k: CFG.get(k) for k in SCIENCE_KEYS}
    d["data_source"] = DATA_SOURCE
    d["dtype"] = "float64"
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_subdir(run_dir, sub):
    p = os.path.join(run_dir, sub)
    os.makedirs(p, exist_ok=True)
    return p


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def write_json_atomic(path, obj):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return _unjson(d)


def _unjson(o):
    if isinstance(o, dict):
        return {k: _unjson(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_unjson(v) for v in o]
    if o in ("nan", "inf", "-inf"):
        return float(o)
    return o


def save_vecs_atomic(path, vecs):
    tmp = path + f".tmp{os.getpid()}"
    torch.save({k: v.detach().to("cpu") for k, v in vecs.items()}, tmp)
    os.replace(tmp, path)


def load_vecs(run_dir, name):
    return torch.load(os.path.join(run_dir, "vecs", safe_name(name) + ".pt"), map_location=DEVICE)


def is_done(run_dir, name, sig):
    p = os.path.join(run_dir, "done", safe_name(name) + ".json")
    v = os.path.join(run_dir, "vecs", safe_name(name) + ".pt")
    if not (os.path.exists(p) and os.path.exists(v)):
        return False
    try:
        return read_json(p).get("signature") == sig
    except Exception:
        return False


# =============================================================================================================
# Starts (built once in the kernel, cached as theta0 files so that workers use the identical theta0)
# =============================================================================================================
def start_paths(run_dir, cname):
    d = run_subdir(run_dir, "starts")
    return os.path.join(d, cname + ".pt"), os.path.join(d, cname + ".json")


def build_or_load_start(cb, run_dir, verbose=True):
    cname = combo_name(cb)
    pt, js = start_paths(run_dir, cname)
    sig = config_signature()
    if os.path.exists(pt) and os.path.exists(js):
        info = read_json(js)
        if info.get("signature") == sig:
            if verbose:
                say(f"  {cname}: loaded cached start (built {info.get('built')})")
            return info
    t1 = time.time()
    prob, th, Y, labels = combo_problem(cb)
    info = dict(combo=cname, act=cb["act"], start=cb["start"], seed=cb["seed"], signature=sig,
                data_source=DATA_SOURCE, built=time.strftime("%Y-%m-%d %H:%M:%S"))
    if cb["start"] == "trained":
        th, fi = fit_dj(prob, th, Y)
        info.update(fi)
    st = Start(prob, th)
    info.update(st.spectrum())
    if not st.ill:
        dirs, dinfo, S0 = make_directions(st, cb["seed"], CFG["N_TRAIN"])
        info.update(directions=dinfo, S0=S0)
    info.update(N=prob.N, P=prob.P, build_s=time.time() - t1)
    torch.save(th.detach().cpu(), pt)
    write_json_atomic(js, info)
    if verbose:
        say(f"  {cname}: built in {fmt_time(time.time() - t1)}")
    del st
    free_memory()
    return info


class Ctx:
    """Everything a combination's cases share: problem, Start (J0, K0 factor, eigen-decomposition), directions."""

    def __init__(self, cb, run_dir):
        self.cb, self.cname = cb, combo_name(cb)
        self.prob, _, self.Y, self.labels = combo_problem(cb)
        pt, _ = start_paths(run_dir, self.cname)
        th0 = torch.load(pt, map_location="cpu").to(DT).to(DEVICE)
        self.st = Start(self.prob, th0, tag=self.cname)
        self.dirs, self.dinfo, self.S0 = make_directions(self.st, cb["seed"], CFG["N_TRAIN"])
        self.run_dir = run_dir


# =============================================================================================================
# Plan
# =============================================================================================================
def L_SL(m):
    return dict(kind="SL", m=m)


def L_DJ(c):
    return dict(kind="DJ", c=c)


def L_GD(c, stop_rel, cap, path=False, dir_stop_rel=None, dir_window=None):
    L = dict(kind="GD", c=c, stop_rel=stop_rel, cap=cap, path=path)
    if dir_stop_rel is not None:                             # A2.1 (added only when used: other cases' L unchanged)
        L.update(dir_stop_rel=dir_stop_rel, dir_window=dir_window)
    return L


def L_HB(c, mu, stop_rel, cap, u_tol_rel=None, dir_stop_rel=None, dir_window=None):
    L = dict(kind="HB", c=c, mu=mu, stop_rel=stop_rel, cap=cap, u_tol_rel=u_tol_rel)
    if dir_stop_rel is not None:                             # A2.1 (added only when used: other cases' L unchanged)
        L.update(dir_stop_rel=dir_stop_rel, dir_window=dir_window)
    return L


def L_SGD(c, B, seed, stop_rel, cap, path=True):
    return dict(kind="SGD", c=c, B=B, seed=seed, stop_rel=stop_rel, cap=cap, path=path)


def L_ADAM(lr, stop_rel, cap):
    return dict(kind="ADAM", lr=lr, stop_rel=stop_rel, cap=cap)


def L_PROX(gt_rel):
    return dict(kind="PROX", gt_rel=gt_rel)


EIG4 = ["eigA", "eigB", "eigC", "eigD"]
DIRS6 = EIG4 + ["label", "rand"]


def _case(cname, grp, L, d, size_lbl, frac, sign, **kw):
    nm = f"{cname}__{grp}__{learner_name(L)}__{d}__{size_lbl}__{'p' if sign > 0 else 'm'}"
    c = dict(name=nm, combo=cname, grp=grp, kind="run", L=L, learner=learner_name(L), dir=d, size=size_lbl,
             frac=frac, sign=sign)
    c.update(kw)
    return c


def plan_combo(cb):
    cname = combo_name(cb)
    s1, s2 = CFG["S1_FRAC"], CFG["S2_FRAC"]
    cases = []
    if cb["act"] == "linear":
        Ls = [L_SL(8), L_DJ(0.5), L_GD(CFG["LIN_GD_C"], CFG["GD_STOP_REL"], CFG["GD_CAP"]),
              L_HB(CFG["LIN_HB_C"], CFG["HB_MU"], CFG["GD_STOP_REL"], CFG["GD_CAP"], CFG["HB_U_TOL_REL"])]
        for d in ("label", "rand"):
            for sg in (1, -1):
                for L in Ls:
                    cases.append(_case(cname, "lin", L, d, "s1", s1, sg))
        return cases
    grid = [(d, "s1", s1) for d in DIRS6] + [(d, "s2", s2) for d in DIRS6]     # A1.1: every small-move case at s1 AND s2
    for d, sl, fr in grid:
        for sg in (1, -1):
            for m in CFG["SL_MS"]:
                cases.append(_case(cname, "sm", L_SL(m), d, sl, fr, sg))
            for c in CFG["DJ_CS"]:
                cases.append(_case(cname, "sm", L_DJ(c), d, sl, fr, sg))
            if d in EIG4:
                dsr, dwin = CFG["DIR_STOP_REL"], CFG["DIR_WINDOW"]  # A2.1: direction-converged stop for H3/H4 cases
                for c in CFG["GD_CS"]:
                    cases.append(_case(cname, "sm", L_GD(c, CFG["GD_STOP_REL"], CFG["GD_CAP"], dir_stop_rel=dsr,
                                                         dir_window=dwin), d, sl, fr, sg))
                for c in CFG["HB_CS"]:                            # A1.1: HB at both sizes
                    cases.append(_case(cname, "sm", L_HB(c, CFG["HB_MU"], CFG["GD_STOP_REL"], CFG["GD_CAP"],
                                                         CFG["HB_U_TOL_REL"], dir_stop_rel=dsr, dir_window=dwin),
                                       d, sl, fr, sg))
            if sl == "s1" and sg > 0:
                for g in CFG["PROX_GT_REL"]:
                    cases.append(_case(cname, "sm", L_PROX(g), d, sl, fr, sg))
    h5L = [L_GD(0.25, CFG["H5_STOP_REL"], CFG["H5_CAP"], path=True),
           L_GD(1.5, CFG["H5_STOP_REL"], CFG["H5_CAP"], path=True),
           L_SGD(CFG["SGD_C"], CFG["SGD_B"], cb["seed"], CFG["H5_STOP_REL"], CFG["H5_CAP"], path=True)]
    for d in ("label", "rand"):
        for sg in (1, -1):
            for L in h5L:
                cases.append(_case(cname, "h5", L, d, "s1", s1, sg))
            cases.append(_case(cname, "r1", L_ADAM(CFG["ADAM_LR"], CFG["R1_STOP_REL"], CFG["R1_CAP"]), d, "s1", s1, sg))
    r2L = [L_SL(32), L_DJ(0.25), L_GD(0.25, CFG["R2_STOP_REL"], CFG["R2_CAP"]),
           L_GD(1.5, CFG["R2_STOP_REL"], CFG["R2_CAP"]), L_HB(0.1, CFG["HB_MU"], CFG["R2_STOP_REL"], CFG["R2_CAP"])]
    for fr in CFG["R2_SIZES"]:
        for L in r2L:
            cases.append(_case(cname, "r2", L, "label", f"r{fr:g}", fr, 1))
    for d in ("label", "rand"):                              # LAW evaluations (need both signs of the h5 runs)
        for L in h5L:
            pn = _case(cname, "h5", L, d, "s1", s1, 1)["name"]
            mn = _case(cname, "h5", L, d, "s1", s1, -1)["name"]
            c = dict(name=f"{cname}__law__{learner_name(L)}__{d}__s1__e", combo=cname, grp="law", kind="law",
                     L=L, learner=learner_name(L), dir=d, size="s1", frac=s1, sign=0, plus=pn, minus=mn)
            cases.append(c)
    only = CFG.get("GRID_ONLY")
    if only:
        cases = [c for c in cases if c["grp"] in only]
    return cases


def all_combos():
    cbs = [dict(act=a, start=s, seed=sd) for a in CFG["ACTS"] for s in CFG["STARTS"] for sd in CFG["SEEDS"]]
    return cbs


def linear_combo():
    return dict(act="linear", start="init", seed=0)


# =============================================================================================================
# Running cases
# =============================================================================================================
def run_case(ctx, case):
    st, prob = ctx.st, ctx.prob
    rec = dict(name=case["name"], combo=ctx.cname, act=ctx.cb["act"], start=ctx.cb["start"], seed=ctx.cb["seed"],
               grp=case["grp"], kind=case["kind"], learner=case["learner"], L=case["L"], dir=case["dir"],
               size=case["size"], frac=case["frac"], sign=case["sign"], S0=ctx.S0, s=case["frac"] * ctx.S0,
               data_source=DATA_SOURCE, lam1_K0=st.lam1)
    di = ctx.dinfo[case["dir"]]
    rec.update(dir_index=di["index"], lam_ratio=di["lam_ratio"])
    rec["f_pred"] = predict_f(case["L"], di["lam_ratio"])
    if case["kind"] == "run":
        s = case["frac"] * ctx.S0
        t = st.F0 + case["sign"] * s * ctx.dirs[case["dir"]]
        np_cap = CFG["NP_CAP_R2"] if case["grp"] == "r2" else None        # A1.6/A2.3: NP cap NP_CAP_R2 for R2 cases only
        sc, vecs = run_and_measure(prob, st, t, s, case["L"], np_cap=np_cap)
        rec.update(sc)
        if "C" in vecs and "C_lin" in vecs:
            rec["Clin_rel_err"] = vnorm(vecs["C"] - vecs["C_lin"]) / max(vnorm(vecs["C"]), 1e-300)
        return rec, vecs
    # LAW case: C^e = (C+ + C-)/2 from the two signs; prediction LAW(C^e) vs measured v_L^e
    vp = load_vecs(ctx.run_dir, case["plus"])
    vm = load_vecs(ctx.run_dir, case["minus"])
    Ce = 0.5 * (vp["C"] + vm["C"])
    vLe = 0.5 * (vp["vL"] + vm["vL"])
    vNe = 0.5 * (vp["vNP"] + vm["vNP"])
    t1 = time.time()
    lw, li = law(prob, st, Ce)
    rec.update(li)
    rec.update(law_cos=cosine(lw, vLe), law_ratio=vnorm(vLe) / vnorm(lw) if vnorm(lw) > 0 else float("nan"),
               law_norm=vnorm(lw), vLe_norm=vnorm(vLe), t_law=time.time() - t1)
    rec.update({"even_" + k: v for k, v in even_stats(vp["vL"], vp["vNP"], vm["vL"], vm["vNP"]).items()})
    return rec, dict(law=lw)


def run_combo_cases(cb, cases, run_dir, sig, worker="kernel"):
    """Run every not-yet-done case of one combination (resumable). Returns (n_run, n_failed)."""
    todo = [c for c in cases if not is_done(run_dir, c["name"], sig)]
    if not todo:
        return 0, 0
    ddir, fdir, vdir = (run_subdir(run_dir, s) for s in ("done", "failed", "vecs"))
    cname = combo_name(cb)
    say(f"[{worker}] {cname}: {len(todo)} of {len(cases)} cases to run ({time.strftime('%H:%M:%S')})")
    ctx = Ctx(cb, run_dir)
    nrun, nfail = 0, 0
    t_start = time.time()
    for i, case in enumerate(todo):
        nm = safe_name(case["name"])
        fp = os.path.join(fdir, nm + ".json")
        if os.path.exists(fp):
            os.remove(fp)
        _HB["label"] = f"[{worker}] {cname} case {i + 1}/{len(todo)} {case['grp']} {case['learner']} {case['dir']} {case['size']} {case['sign']:+d}:"
        heartbeat()
        t1 = time.time()
        try:
            test_fail_hook(case["name"])
            rec, vecs = run_case(ctx, case)
            rec.update(signature=sig, worker=worker, device=str(DEVICE), wall_s=time.time() - t1,
                       finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            save_vecs_atomic(os.path.join(vdir, nm + ".pt"), vecs)
            write_json_atomic(os.path.join(ddir, nm + ".json"), rec)
            nrun += 1
        except Exception:
            tb = traceback.format_exc()
            say(f"\n!!!!! [{worker}] CASE FAILED: {case['name']}\n{tb}")
            write_json_atomic(fp, dict(name=case["name"], combo=cname, worker=worker, error=tb,
                                       time=time.strftime("%Y-%m-%d %H:%M:%S")))
            nfail += 1
        free_memory()
    say(f"[{worker}] {cname}: finished {nrun} cases, {nfail} failed, in {fmt_time(time.time() - t_start)}")
    return nrun, nfail


# =============================================================================================================
# Workers (one subprocess per GPU, static split of the combinations by index modulo the number of workers)
# =============================================================================================================
def write_run_files(run_dir, plan):
    cfg = dict(CFG)
    cfg.update(FORCE_DATA_SOURCE=DATA_SOURCE, MNIST_DOWNLOAD=False,
               MNIST_ROOT=os.path.abspath(CFG.get("MNIST_ROOT", "./mnist_data")), SIGNATURE=config_signature())
    write_json_atomic(os.path.join(run_dir, "config.json"), cfg)
    write_json_atomic(os.path.join(run_dir, "plan.json"), plan)


def worker_main(run_dir, k, n, device, threads):
    cfg = read_json(os.path.join(run_dir, "config.json"))
    configure(cfg)
    torch.set_num_threads(int(threads))
    set_device("cpu" if device == "cpu" else "cuda")
    plan = read_json(os.path.join(run_dir, "plan.json"))
    wid = f"w{k}-{'cpu' if device == 'cpu' else 'gpu' + str(device)}"
    say(f"[{wid}] pid {os.getpid()} device {DEVICE} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"threads {torch.get_num_threads()} ({time.strftime('%H:%M:%S')})")
    _load_mnist_raw()
    sig = config_signature()
    assert sig == cfg["SIGNATURE"], f"worker signature {sig} != kernel signature {cfg['SIGNATURE']}"
    for i, item in enumerate(plan):
        if i % n != k:
            continue
        run_combo_cases(item["combo"], item["cases"], run_dir, sig, worker=wid)
    say(f"[{wid}] finished ({time.strftime('%H:%M:%S')})")


def _spawn(run_dir, k, n, dev, threads):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "" if dev == "cpu" else str(dev)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = str(threads)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONIOENCODING"] = "utf-8"
    libdir = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-u", "-c",
           f"import sys; sys.path.insert(0, {libdir!r}); import part8lib; "
           f"part8lib.worker_main({run_dir!r}, {k}, {n}, {dev!r}, {int(threads)})"]
    logp = os.path.join(run_subdir(run_dir, "logs"), f"worker{k}.log")
    log = open(logp, "a", encoding="utf-8")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
    return dict(p=p, log=log, logp=logp, dev=dev)


def _tail(path, n=20):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""


def parallel_slots():
    """Worker devices: one per GPU if >= 2 GPUs; CPU workers only when asked for (testing)."""
    ng = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cpu_w = int(os.environ.get("PART8_CPU_WORKERS", CFG.get("CPU_WORKERS", 0)) or 0)
    if CFG.get("PARALLEL", True) and ng >= 2:
        env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ids = [x.strip() for x in env.split(",") if x.strip()] if env.strip() else [str(i) for i in range(ng)]
        return ids[:ng], f"{ng} GPUs: one worker per GPU"
    if cpu_w > 0:
        return ["cpu"] * cpu_w, f"{cpu_w} CPU worker processes (testing mode)"
    return [], f"{ng} GPU(s): in-kernel path"


def run_pool(plan, run_dir, slots):
    write_run_files(run_dir, plan)
    sig = config_signature()
    threads = torch.get_num_threads()   # CPU workers use the kernel's thread count (identical BLAS arithmetic)
    n = len(slots)
    procs = [_spawn(run_dir, k, n, slots[k], threads) for k in range(n)]
    for k, P in enumerate(procs):
        mine = [combo_name(it["combo"]) for i, it in enumerate(plan) if i % n == k]
        say(f"  worker {k} on {'CPU' if P['dev'] == 'cpu' else 'GPU ' + str(P['dev'])}: pid {P['p'].pid}, "
            f"combinations {mine}, log {P['logp']}")
    names = [c["name"] for it in plan for c in it["cases"]]
    t0, last = time.time(), time.time()
    try:
        while True:
            time.sleep(2.0)
            alive = [P for P in procs if P["p"].poll() is None]
            if time.time() - last >= CFG.get("PROGRESS_EVERY_S", 60) or not alive:
                last = time.time()
                nd = sum(is_done(run_dir, nm, sig) for nm in names)
                nf = sum(os.path.exists(os.path.join(run_dir, "failed", safe_name(nm) + ".json")) for nm in names)
                say(f"  [{time.strftime('%H:%M:%S')}] progress: {nd}/{len(names)} cases done, {nf} failed, "
                    f"{len(alive)}/{n} workers running, elapsed {fmt_time(time.time() - t0)}")
            if not alive:
                break
    except KeyboardInterrupt:
        say("interrupted: terminating workers (finished cases stay on disk; re-run to resume)")
        for P in procs:
            if P["p"].poll() is None:
                P["p"].terminate()
        raise
    finally:
        for P in procs:
            P["log"].close()
    for k, P in enumerate(procs):
        if P["p"].returncode != 0:
            say(f"!!! worker {k} exited with code {P['p'].returncode}; last log lines:\n{_tail(P['logp'])}")


def run_grid(plan, run_dir):
    sig = config_signature()
    slots, why = parallel_slots()
    say(f"execution: {why}")
    kern = [it for it in plan if it.get("kernel_only")]
    rest = [it for it in plan if not it.get("kernel_only")]
    for it in kern:
        run_combo_cases(it["combo"], it["cases"], run_dir, sig)
    if slots and rest:
        run_pool(rest, run_dir, slots)
    else:
        for it in rest:
            run_combo_cases(it["combo"], it["cases"], run_dir, sig)


def collect(plan, run_dir):
    """All done records of the plan (current signature) and the failure list."""
    sig = config_signature()
    recs, failed, missing = {}, [], []
    for it in plan:
        for c in it["cases"]:
            nm = safe_name(c["name"])
            p = os.path.join(run_dir, "done", nm + ".json")
            fpath = os.path.join(run_dir, "failed", nm + ".json")
            if is_done(run_dir, c["name"], sig):
                recs[c["name"]] = read_json(p)
            elif os.path.exists(fpath):
                failed.append(read_json(fpath))
            else:
                missing.append(c["name"])
    return recs, failed, missing

# =============================================================================================================
# Unit tests (spec sec. 6). Each returns (passed, message). Run by the TESTS cell through run_tests().
# =============================================================================================================
def _mlp_init_start(act, seed=0):
    X, labels, Y = get_data(seed, CFG["N_TRAIN"])
    prob, th = layered_problem(X, [196, CFG["WIDTH"], CFG["WIDTH"], CFG["N_CLASSES"]], act, seed, name=f"{act}-init")
    return prob, Start(prob, th)


def _u6_setup():
    """U6 network: 8 -> 32 -> 32 -> 1 tanh, n = 16 Gaussian inputs, seed 0, start = DJ(0.5) fit of Gaussian targets."""
    if "u6" in _TEST_CACHE:
        return _TEST_CACHE["u6"]
    X = torch.randn(16, 8, generator=gen(100), dtype=DT).to(DEVICE)
    tfit = torch.randn(16, generator=gen(101), dtype=DT).to(DEVICE)
    prob, th = layered_problem(X, [8, 32, 32, 1], "tanh", 0, name="U6")
    th, fi = fit_dj(prob, th, tfit)
    st = Start(prob, th)
    rv = torch.randn(16, generator=gen(102), dtype=DT).to(DEVICE)
    out = dict(prob=prob, st=st, tfit=tfit, fit=fi, q1=st.Q[:, 0].clone(), rand=rv / vnorm(rv),
               s=0.05 * vnorm(tfit))
    _TEST_CACHE["u6"] = out
    return out


_TEST_CACHE = {}


def t_U1():
    msgs, ok = [], True
    for act in ("tanh", "gelu", "relu"):
        prob, th = layered_problem(get_data(0, CFG["N_TRAIN"])[0],
                                   [196, CFG["WIDTH"], CFG["WIDTH"], CFG["N_CLASSES"]], act, 0)
        J = prob.J(th)
        g = gen(4242)
        rows = torch.randint(0, prob.N, (20,), generator=g)
        cols = torch.randint(0, prob.P, (20,), generator=g)
        h = 1e-5
        fd, an = [], []
        for a, p in zip(rows.tolist(), cols.tolist()):
            e = torch.zeros(prob.P, dtype=DT, device=th.device)
            e[p] = h
            fd.append(float((prob.F(th + e)[a] - prob.F(th - e)[a]) / (2 * h)))
            an.append(float(J[a, p]))
        fd, an = np.array(fd), np.array(an)
        rel = float(np.linalg.norm(fd - an) / max(np.linalg.norm(an), 1e-300))
        ok_a = rel <= 1e-6
        ok &= ok_a
        # supplementary engineering check (not in the spec): explicit back-propagation used by GD/HB/SGD/ADAM == J^T r
        r = torch.randn(prob.N, generator=gen(4243), dtype=DT).to(th.device)
        _, cache = prob.fwd(th)
        vj = prob.vjp(cache, r)
        rel_b = vnorm(vj - J.T @ r) / vnorm(J.T @ r)
        ok &= rel_b <= 1e-12
        msgs.append(f"{act}: FD rel err (20 entries, norm-wise) {rel:.2e} [<=1e-6]; backprop vs J^T r {rel_b:.1e} [<=1e-12]")
    return ok, " | ".join(msgs)


def t_U2():
    prob, st = _mlp_init_start(CFG["ACTS"][0])
    worst1 = 0.0
    for k in range(5):
        e = torch.randn(prob.N, generator=gen(4300 + k), dtype=DT).to(st.th0.device)
        x = st.J0.T @ e
        worst1 = max(worst1, vnorm(st.V0(x)) / vnorm(x))
    w = torch.randn(prob.P, generator=gen(4310), dtype=DT).to(st.th0.device)
    v = st.V0(w)
    idem = vnorm(st.V0(v) - v) / vnorm(v)
    return worst1 <= 1e-9 and idem <= 1e-9, f"max |V0 J0^T e|/|J0^T e| = {worst1:.2e} [<=1e-9]; |V0V0w - V0w|/|V0w| = {idem:.2e} [<=1e-9]"


def t_U3():
    prob, st = _mlp_init_start(CFG["ACTS"][0])
    dirs, _, S0 = make_directions(st, 0, CFG["N_TRAIN"])
    s = CFG["S1_FRAC"] * S0
    out, ok = [], True
    thb, Jb = st.th0, st.J0
    for nm, t in (("theta0 -> F0+s1*rand", st.F0 + s * dirs["rand"]), ("2nd base -> F0+s1*label", st.F0 + s * dirs["label"])):
        jr = jump(prob, thb, t, Jb=Jb)
        dth = jr["theta"] - thb
        Lb = chol(gram(Jb))
        vb = dth - Jb.T @ csolve(Lb, Jb @ dth)
        rel = vnorm(vb) / vnorm(dth)
        ok &= jr["status"] == "CONVERGED" and jr["res_inf"] <= 1e-11 and rel <= 1e-9
        out.append(f"{nm}: residual {jr['res_inf']:.1e} [<=1e-11], |V_b dtheta|/|dtheta| {rel:.1e} [<=1e-9], "
                   f"chord its {jr['it_chord']}, newton {jr['it_newton']}")
        thb = jr["theta"]
        Jb = prob.J(thb)
    return ok, " | ".join(out)


def t_U4():
    prob, st = _mlp_init_start(CFG["ACTS"][0])
    dirs, _, S0 = make_directions(st, 0, CFG["N_TRAIN"])
    t = st.F0 + CFG["S1_FRAC"] * S0 * dirs["label"]
    dj = run_DJ(prob, st, t, 0.5)
    FL = prob.F(dj["theta"])
    npr = nearest_point(prob, st.th0, FL, dj["theta"])
    th_ = jump(prob, st.th0, FL, Jb=st.J0)
    sl = run_SL(prob, st, FL, 8)
    dNP = vnorm(npr["theta"] - st.th0)
    dTH, dSL, dDJ = (vnorm(x - st.th0) for x in (th_["theta"], sl["theta"], dj["theta"]))
    ok = (npr["status"] == "CONVERGED" and npr["res"] <= 1e-10 and npr["stat"] <= 1e-8
          and dNP <= dTH and dNP <= dSL and dNP <= dDJ)
    return ok, (f"NP residual {npr['res']:.1e} [<=1e-10], stat {npr['stat']:.1e} [<=1e-8], its {npr['it']}; "
                f"|NP-th0| = {dNP:.10e} <= TH {dTH:.10e}, SL(8) {dSL:.10e}, DJ(0.5) {dDJ:.10e}")


U5_CASES = [("SL(2)", lambda: L_SL(2), 0.25), ("SL(8)", lambda: L_SL(8), 0.4375), ("DJ(0.5)", lambda: L_DJ(0.5), 1 / 3),
            ("DJ(1.5)", lambda: L_DJ(1.5), -1.0), ("DJ(1.9)", lambda: L_DJ(1.9), -9.0)]


def t_U5():
    prob, th0 = toy_problem()
    st = Start(prob, th0)
    d = torch.ones(1, dtype=DT, device=th0.device)
    s = 0.02
    gs, gc_ = CFG["GD_STOP_REL"], CFG["GD_CAP"]
    cases = [(L_SL(2), 0.25), (L_SL(8), 0.4375), (L_DJ(0.5), 0.3333), (L_DJ(1.5), -1.0), (L_DJ(1.9), -9.0),
             (L_GD(0.5, gs, gc_), 0.3333), (L_GD(1.5, gs, gc_), -1.0),
             (L_HB(0.05, 0.9, gs, gc_, CFG["HB_U_TOL_REL"]), -1.9067),
             (L_HB(0.0105, 0.9, gs, gc_, CFG["HB_U_TOL_REL"]), 0.0),
             (L_PROX(1e-2), 1.0), (L_PROX(1e-3), 1.0)]
    ok, parts = True, []
    for L, val in cases:
        ev, scs, _ = run_pair(prob, st, d, s, L)
        good = abs(ev["rho"] - val) <= 0.01 * max(1.0, abs(val))
        good &= all(x["learner_status"] == "CONVERGED" and x["np_status"] == "CONVERGED" for x in scs)
        ok &= good
        parts.append(f"{learner_name(L)} rho={ev['rho']:+.4f} (want {val:+.4f}, cos {ev['cos']:+.4f}) {'ok' if good else 'FAIL'}")
    return ok, "even part, s=0.02, tol 0.01*max(1,|value|): " + "; ".join(parts)


def t_U6():
    u = _u6_setup()
    prob, st, s = u["prob"], u["st"], u["s"]
    ok, parts = True, []
    checks = [(L_SL(8), "q1", 0.4375, 0.01), (L_SL(8), "rand", 0.4375, 0.01),
              (L_DJ(1.5), "q1", -1.0, 0.02), (L_DJ(1.5), "rand", -1.0, 0.02),
              (L_GD(1.9, CFG["GD_STOP_REL"], CFG["GD_CAP"]), "q1", -9.0, 0.1)]
    for L, dn, val, tol in checks:
        ev, scs, _ = run_pair(prob, st, u[dn], s, L)
        good = abs(ev["rho"] - val) <= tol and all(x["np_status"] == "CONVERGED" for x in scs)
        ok &= good
        parts.append(f"{learner_name(L)} {dn}: rho={ev['rho']:+.4f} (want {val:+.4f} +-{tol}, cos {ev['cos']:+.4f}, "
                     f"steps {scs[0]['steps']}/{scs[1]['steps']} {scs[0]['learner_status']}) {'ok' if good else 'FAIL'}")
    return ok, (f"fit: {u['fit']['fit_iterations']} its, mse {u['fit']['fit_mse']:.1e}; lam1={st.lam1:.3e}, "
                f"cond={st.lam1 / st.lamN:.2e}, s={s:.4f}: " + "; ".join(parts))


def t_U7():
    X = torch.randn(3, 4, generator=gen(110), dtype=DT).to(DEVICE)
    prob, th = layered_problem(X, [4, 4], "tanh", 0, name="lin20")
    assert prob.P == 20
    st = Start(prob, th)
    worst = 0.0
    for k, dseed in enumerate((111, 112)):
        dv = torch.randn(prob.N, generator=gen(dseed), dtype=DT).to(DEVICE)
        dv = dv / vnorm(dv)
        s = 0.1
        for L in (L_SL(8), L_DJ(0.5), L_GD(1.5, CFG["GD_STOP_REL"], CFG["GD_CAP"]),
                  L_HB(0.1, CFG["HB_MU"], CFG["GD_STOP_REL"], CFG["GD_CAP"], CFG["HB_U_TOL_REL"])):
            for sg in (1, -1):
                sc, _ = run_and_measure(prob, st, st.F0 + sg * s * dv, s, L)
                worst = max(worst, sc["vL_over_dL"], sc["vNP_over_dNP"])
    return worst <= 1e-9, f"20-parameter linear model (4->4, n=3): max over 16 runs of |v_L|/|dL|, |v_NP|/|dNP| = {worst:.2e} [<=1e-9]"


def t_U8():
    cb = linear_combo()
    prob, th, Y, labels = combo_problem(cb)
    st = Start(prob, th)
    dirs, _, S0 = make_directions(st, 0, CFG["N_TRAIN"])
    t = st.F0 + 1.0 * dirs["rand"]
    c = 0.7
    one = run_first_order(prob, st, t, 1.0, L_GD(c, 0.0, 1))
    assert one["steps"] == 1
    r0 = st.F0 - t
    r1 = prob.F(one["theta"]) - t
    worst, parts = 0.0, []
    for i in (1, 2, 5, 11, prob.N):                         # 1, 2, 5 = spec; 11 and N = supplementary (other eigenvalues)
        q = st.Q[:, i - 1]
        want = (1 - c * float(st.lam[i - 1]) / st.lam1) * float(q @ r0)
        got = float(q @ r1)
        rel = abs(got - want) / abs(want)
        worst = max(worst, rel)
        parts.append(f"i={i}: {rel:.1e}")
    mu = CFG["HB_MU"]
    rs = [float(st.Q[:, 0] @ r0)]
    for k in range(1, 6):
        hb = run_first_order(prob, st, t, 1.0, L_HB(c, mu, 0.0, k))
        rs.append(float(st.Q[:, 0] @ (prob.F(hb["theta"]) - t)))
    worst_hb = 0.0
    for k in range(5):
        prev = rs[k - 1] if k > 0 else rs[0]                # r_{-1} = r_0
        pred = (1 + mu - c) * rs[k] - mu * prev
        worst_hb = max(worst_hb, abs(rs[k + 1] - pred) / max(abs(rs[k + 1]), abs(rs[k]), abs(prev)))
    return (worst <= 1e-9 and worst_hb <= 1e-9,
            f"linear control, one GD(0.7) step, rel err of mode factor (1 - c lam_i/lam1): {', '.join(parts)} [<=1e-9]; "
            f"HB(0.7, {mu}) recurrence along q1 over 5 steps: max rel err {worst_hb:.1e} [<=1e-9]")


def t_U9():
    prob, st = _mlp_init_start(CFG["ACTS"][0])
    dirs, _, S0 = make_directions(st, 0, CFG["N_TRAIN"])
    s = CFG["S1_FRAC"] * S0
    t = st.F0 + s * dirs["label"]
    errs = []
    for L in (L_GD(0.25, 0.0, 100, path=True), L_SGD(CFG["SGD_C"], max(1, CFG["SGD_B"] // 2), 0, 0.0, 100, path=True)):
        out = run_first_order(prob, st, t, s, L)
        errs.append((learner_name(L), out["path_identity_err"], out["steps"]))
    ok = all(e <= 1e-9 for _, e, _ in errs)
    return ok, "; ".join(f"{n} ({k} steps): |C+C^T - (dY dY^T - Q)|/|dY dY^T - Q| = {e:.1e} [<=1e-9]" for n, e, k in errs)


def t_U10():
    u = _u6_setup()
    prob, st, s = u["prob"], u["st"], u["s"]
    ok, parts = True, []
    for dn in ("q1", "rand"):
        d = u[dn]
        vs = []
        for sg in (1, -1):
            npr = nearest_point(prob, st.th0, st.F0 + sg * s * d, st.th0)      # TH-type reference: start at theta0
            assert npr["status"] == "CONVERGED"
            vs.append(st.V0(npr["theta"] - st.th0))
        vNe = 0.5 * (vs[0] + vs[1])
        lw, li = law(prob, st, s * s * torch.outer(d, d))
        cs = cosine(lw, vNe)
        ratio = vnorm(vNe) / vnorm(lw)
        good = cs >= 0.999 and 0.97 <= ratio <= 1.03
        ok &= good
        parts.append(f"{dn}: cos(LAW(dY dY^T), v_NP^e) = {cs:.6f} [>=0.999], |v_NP^e|/|LAW| = {ratio:.5f} [0.97,1.03]")
    return ok, "; ".join(parts)


def t_U11():
    u = _u6_setup()
    prob, st = u["prob"], u["st"]
    p, q = st.Q[:, 1].clone(), st.Q[:, 11].clone()           # eigenvectors 2 and 12 (1-based)
    eps = 0.05

    def loop(a, b):
        th, F_prev = st.th0.clone(), st.F0.clone()
        Cm = torch.zeros(prob.N, prob.N, dtype=DT, device=th.device)
        for tg in (st.F0 + eps * a, st.F0 + eps * a + eps * b, st.F0 + eps * b, st.F0.clone()):
            jr = jump(prob, th, tg)
            assert jr["status"] == "CONVERGED"
            th = jr["theta"]
            Fn = prob.F(th)
            Cm += torch.outer(Fn - F_prev, F_prev - st.F0)
            F_prev = Fn
        return th, Cm

    thA, CA = loop(p, q)
    thB, CB = loop(q, p)
    D = CA - CB
    Dw = 2 * eps * eps * (torch.outer(q, p) - torch.outer(p, q))
    relC = vnorm(D - Dw) / vnorm(Dw)
    w = st.V0(thA - thB)
    l1, _ = law(prob, st, D)
    l2, _ = law(prob, st, D.T)
    c1, c2 = cosine(w, l1), cosine(w, l2)
    ok = relC <= 1e-9 and c1 >= 0.99 and c2 <= -0.99
    return ok, (f"|C_A - C_B - 2 eps^2 (q p^T - p q^T)|/|.| = {relC:.1e} [<=1e-9]; cos(V0 w, LAW(C_A-C_B)) = {c1:+.5f} "
                f"[>=0.99]; cos(V0 w, LAW((C_A-C_B)^T)) = {c2:+.5f} [<=-0.99]; |V0 w| = {vnorm(w):.3e}")


def t_U12():
    prob, st = _mlp_init_start(CFG["ACTS"][0])
    dirs, _, S0 = make_directions(st, 0, CFG["N_TRAIN"])
    s = CFG["S1_FRAC"] * S0
    r = []
    for _ in range(2):
        ev, _, _ = run_pair(prob, st, dirs["eigA"], s, L_SL(2))
        r.append(ev["rho"])
    diff = abs(r[0] - r[1])
    return diff <= 1e-10, f"H1 case SL(2) eigA s1 (even part) twice: rho = {r[0]:.15f}, {r[1]:.15f}, |diff| = {diff:.1e} [<=1e-10]"


def t_U13():
    u = _u6_setup()
    prob, st = u["prob"], u["st"]
    g = gen(130)
    X = torch.randn(prob.N, 3, generator=g, dtype=DT).to(DEVICE)
    Yv = torch.randn(prob.N, 3, generator=g, dtype=DT).to(DEVICE)
    Cm = 0.0025 * X @ Yv.T
    a, info = law(prob, st, Cm)
    b = law_columns(prob, st, Cm)
    rel = vnorm(a - b) / vnorm(b)
    return rel <= 1e-6, f"random rank-3 C: |LAW_svd - LAW_columns|/|LAW_columns| = {rel:.2e} [<=1e-6] (kept {info['kept']} singular values)"


def t_UW(run_dir):
    """Worker identity: one case in the kernel and in a worker subprocess; every stored number must be identical."""
    slots, why = parallel_slots()
    ng = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ng >= 1:
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    elif slots:
        dev = "cpu"
    else:
        return None, "skipped (no GPU and no CPU workers requested)"
    cb = dict(act=CFG["ACTS"][0], start="init", seed=0)
    case = [c for c in plan_combo_unfiltered(cb) if c["grp"] == "sm" and c["learner"] == "SL(2)"
            and c["dir"] == "label" and c["size"] == "s1" and c["sign"] == 1][0]
    base = os.path.join(run_dir, "_worker_identity_test")
    if os.path.exists(base):
        shutil.rmtree(base)
    dK, dW = os.path.join(base, "kernel"), os.path.join(base, "worker")
    build_or_load_start(cb, dK, verbose=False)
    run_subdir(dW, "starts")
    for f in glob.glob(os.path.join(dK, "starts", "*")):
        shutil.copy(f, os.path.join(dW, "starts"))
    sig = config_signature()
    run_combo_cases(cb, [case], dK, sig, worker="kernel-test")
    plan = [dict(combo=cb, cases=[case])]
    write_run_files(dW, plan)
    P = _spawn(dW, 0, 1, dev, torch.get_num_threads())
    P["p"].wait()
    P["log"].close()
    nm = safe_name(case["name"])
    try:
        rk = read_json(os.path.join(dK, "done", nm + ".json"))
        rw = read_json(os.path.join(dW, "done", nm + ".json"))
    except Exception as e:
        return False, f"worker produced no result ({e!r}); log:\n{_tail(P['logp'])}"
    skip = {"worker", "device", "wall_s", "finished", "t_learner", "t_measure"}
    diffs = [k for k in rk if k not in skip and rk[k] != rw.get(k) and not (isinstance(rk[k], float) and math.isnan(rk[k])
                                                                         and math.isnan(rw.get(k, 0.0)))]
    vk = torch.load(os.path.join(dK, "vecs", nm + ".pt"))
    vw = torch.load(os.path.join(dW, "vecs", nm + ".pt"))
    vdiff = [k for k in vk if not torch.equal(vk[k], vw[k])]
    ok = not diffs and not vdiff
    return ok, (f"case {case['name']} on {'CPU' if dev == 'cpu' else 'GPU ' + dev}: rho kernel {rk['rho']!r} worker "
                f"{rw['rho']!r}; differing scalars: {diffs or 'none'}; differing vectors: {vdiff or 'none'}")


def plan_combo_unfiltered(cb):
    keep = CFG.get("GRID_ONLY")
    CFG["GRID_ONLY"] = None
    try:
        return plan_combo(cb)
    finally:
        CFG["GRID_ONLY"] = keep


def t_U14():
    """A2.2: ill-conditioned start (U6 network and U6 weights, inputs 14, 15 replaced by near-copies of inputs 0, 1);
    GD(1.0), HB(0.1, 0.9)
    toward q_2 at s = 0.05 ||t_fit||: old stop rule at a 20 000-step cap -> UNCONVERGED; A2.1 rule -> CONVERGED in
    < 20 000 steps with |rho - f| <= 0.03 max(1, |f|) and |cos| >= 0.99 (even part)."""
    pert = CFG["U14_PERT"]
    X = torch.randn(16, 8, generator=gen(100), dtype=DT)
    tfit = torch.randn(16, generator=gen(101), dtype=DT).to(DEVICE)
    noise = torch.randn(2, 8, generator=gen(140), dtype=DT)
    X[14] = X[0] + pert * noise[0]
    X[15] = X[1] + pert * noise[1]
    prob, _ = layered_problem(X.to(DEVICE), [8, 32, 32, 1], "tanh", 0, name="U14")
    th = _u6_setup()["st"].th0.clone()            # theta0 = the U6 start (fitted on the original inputs); only the
    fi = _u6_setup()["fit"]                       # inputs change (a start only has to be a fixed point in weight space)
    st = Start(prob, th)
    cond = st.lam1 / st.lamN
    q2 = st.Q[:, 1].clone()
    s = 0.05 * vnorm(tfit)
    ratio = float(st.lam[1]) / st.lam1
    cap = 20000
    ok = cond >= 1e4
    parts = [f"U6 weights, inputs 14/15 = inputs 0/1 + {pert:g}*noise: cond(K0) = {cond:.3e} [>=1e4], "
             f"lam2/lam1 = {ratio:.4f}, s = {s:.4f}"]
    for old, new in ((L_GD(1.0, CFG["GD_STOP_REL"], cap), L_GD(1.0, CFG["GD_STOP_REL"], cap, dir_stop_rel=CFG["DIR_STOP_REL"],
                                                                 dir_window=CFG["DIR_WINDOW"])),
                     (L_HB(0.1, 0.9, CFG["GD_STOP_REL"], cap, CFG["HB_U_TOL_REL"]),
                      L_HB(0.1, 0.9, CFG["GD_STOP_REL"], cap, CFG["HB_U_TOL_REL"], dir_stop_rel=CFG["DIR_STOP_REL"],
                           dir_window=CFG["DIR_WINDOW"]))):
        stat_old = [run_first_order(prob, st, st.F0 + sg * s * q2, s, old)["status"] for sg in (1, -1)]
        a_ok = all(x == "UNCONVERGED" for x in stat_old)
        ev, scs, _ = run_pair(prob, st, q2, s, new)
        f = predict_f(new, ratio)
        b_ok = (all(x["learner_status"] == "CONVERGED" and x["steps"] < cap for x in scs)
                and abs(ev["rho"] - f) <= 0.03 * max(1.0, abs(f)) and abs(ev["cos"]) >= 0.99)
        ok &= a_ok and b_ok
        parts.append(f"{learner_name(new)}: (a) old rule, cap {cap}: {'/'.join(stat_old)} {'ok' if a_ok else 'FAIL'}; "
                     f"(b) A2.1 rule: {scs[0]['learner_status']}/{scs[1]['learner_status']} in {scs[0]['steps']}/{scs[1]['steps']} "
                     f"steps, rho = {ev['rho']:+.4f} (f = {f:+.4f}, tol {0.03 * max(1.0, abs(f)):.3f}), cos = {ev['cos']:+.4f}, "
                     f"|F-t|/s at stop {scs[0]['learner_final_res_over_s']:.2e}/{scs[1]['learner_final_res_over_s']:.2e}, "
                     f"|u| {scs[0]['learner_u_norm']:.1e}/{scs[1]['learner_u_norm']:.1e} {'ok' if b_ok else 'FAIL'}")
    return ok, " | ".join(parts)


TESTS = [("U1 Jacobian vs finite differences (each activation)", t_U1), ("U2 projector", t_U2), ("U3 JUMP", t_U3),
         ("U4 nearest point", t_U4), ("U5 exactly solvable model u^2 - v^2", t_U5),
         ("U6 small tanh network", t_U6), ("U7 linear control (20 parameters)", t_U7),
         ("U8 step-factor mapping", t_U8), ("U9 path bookkeeping identity", t_U9),
         ("U10 LAW on a straight path", t_U10), ("U11 LAW index order", t_U11), ("U12 determinism", t_U12),
         ("U13 SVD truncation in LAW", t_U13), ("U14 direction-converged stop rule (A2.2)", t_U14)]


def run_tests(run_dir):
    results = []
    t_all = time.time()
    for name, fn in TESTS + [("UW worker identity (kernel vs worker subprocess)", lambda: t_UW(run_dir))]:
        t1 = time.time()
        try:
            ok, msg = fn()
        except Exception:
            ok, msg = False, "EXCEPTION:\n" + traceback.format_exc()
        tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        say(f"{tag}  {name}  ({time.time() - t1:.1f}s)\n      {msg}")
        results.append(dict(test=name, result=tag, detail=msg, seconds=time.time() - t1))
    say(f"TESTS finished in {fmt_time(time.time() - t_all)}")
    return results

# =============================================================================================================
# Analysis: H-table (pre-registered rules) and R-tables (report only)
# =============================================================================================================
def matches(rho, cos, nr, f, tol):
    """Spec sec. 5: |rho - f| <= tol max(1,|f|) AND (|f| >= 0.05: cos sign(f) >= 0.99; else nr <= 0.05 + tol)."""
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (rho, f)):
        return False
    if abs(rho - f) > tol * max(1.0, abs(f)):
        return False
    if abs(f) >= CFG["F_SMALL"]:
        return isinstance(cos, float) and math.isfinite(cos) and cos * math.copysign(1.0, f) >= CFG["COS_MATCH"]
    return isinstance(nr, float) and math.isfinite(nr) and nr <= CFG["NR_SMALL"] + tol


def _fmt(x, w=9, p=4):
    if x is None:
        return " " * (w - 1) + "-"
    if isinstance(x, str):
        return x.rjust(w)
    if not math.isfinite(x):
        return "nan".rjust(w)
    if x != 0 and (abs(x) < 10 ** (-p + 1) or abs(x) >= 1e5):
        return f"{x:.{max(p - 2, 1)}e}".rjust(w)
    return f"{x:.{p}f}".rjust(w)


def _med(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


class Analysis:
    def __init__(self, plan, run_dir, start_infos):
        self.plan, self.run_dir, self.si = plan, run_dir, start_infos
        self.recs, self.failed, self.missing = collect(plan, run_dir)
        self.planned = {c["name"]: c for it in plan for c in it["cases"]}
        self._vec = {}
        self.lines = []
        self._need_conv = False

    def p(self, s=""):
        self.lines.append(s)
        say(s)

    def vec(self, name):
        if name not in self._vec:
            self._vec[name] = {k: v.to("cpu") for k, v in load_vecs(self.run_dir, name).items()}
        return self._vec[name]

    def pairs(self, group, grp, pred):
        """Even-part pairs: dict key -> dict(plus, minus, even, reason). Planned pairs without a record are listed."""
        out = []
        names = [n for n, c in self.planned.items() if c["grp"] == grp and c["kind"] == "run" and c["sign"] == 1
                 and self._group_of(c) == group and pred(c)]
        for n in sorted(names):
            mn = n[:-1] + "m"
            rp, rm = self.recs.get(n), self.recs.get(mn)
            c = self.planned[n]
            item = dict(name=n[:-3], learner=c["learner"], dir=c["dir"], size=c["size"], combo=c["combo"], plus=rp, minus=rm)
            if rp is None or rm is None:
                item["reason"] = "MISSING/FAILED"
            else:
                item["reason"] = usable(rp, self._need_conv) or usable(rm, self._need_conv)
                if "vL" in self.vec(n) and "vL" in self.vec(mn):
                    vp, vm = self.vec(n), self.vec(mn)
                    item["even"] = even_stats(vp["vL"], vp["vNP"], vm["vL"], vm["vNP"])
                item["f"] = rp.get("f_pred")
            out.append(item)
        return out

    def _group_of(self, c):
        a, s, _ = c["combo"].rsplit("-", 2)
        return (a, s)

    def rule_match(self, hname, group, learners, tol, dirs=None, need_conv=False, conv_frac=None, label=""):
        self._need_conv = need_conv
        items = self.pairs(group, "sm", lambda c: c["learner"] in learners and (dirs is None or c["dir"] in dirs))
        self._need_conv = False
        used = [x for x in items if not x["reason"] and "even" in x]
        excl = [x for x in items if x["reason"] or "even" not in x]
        for x in used:
            e = x["even"]
            x["match"] = matches(e["rho"], e["cos"], e["nr"], x["f"], tol)
        nm = sum(x["match"] for x in used)
        frac = nm / len(used) if used else float("nan")
        verdict = "NO DATA" if not used else ("PASS" if frac >= CFG["RULE_FRAC"] else "FAIL")
        extra = ""
        if conv_frac is not None:
            nconv = sum(1 for x in items if x["plus"] is not None and x["minus"] is not None
                        and x["plus"].get("learner_status") == "CONVERGED" and x["minus"].get("learner_status") == "CONVERGED")
            cf = nconv / len(items) if items else float("nan")
            extra = f"; converged {nconv}/{len(items)} planned = {100 * cf:.0f}% [rule >= {100 * conv_frac:.0f}%]"
            if used and cf < conv_frac:
                verdict = "FAIL"
        self.p(f"{hname} [{label}] {group[0]}/{group[1]}: matched {nm}/{len(used)} = {100 * frac:.1f}% [rule >= "
               f"{100 * CFG['RULE_FRAC']:.0f}%, tol {tol}]{extra} -> {verdict}")
        self._print_excl(excl)
        bad = [x for x in used if not x["match"]]
        if bad:
            self.p("    failing cases (even part):  learner      dir   size       rho         f       cos        nr")
            for x in bad:
                e = x["even"]
                self.p(f"      {x['combo']:18s} {x['learner']:12s} {x['dir']:5s} {x['size']:4s} {_fmt(e['rho'])} "
                       f"{_fmt(x['f'])} {_fmt(e['cos'])} {_fmt(e['nr'])}")
        self._learner_table(used)
        return dict(rule=hname, label=label, group=group, matched=nm, used=len(used), frac=frac, verdict=verdict,
                    excluded=[(x["name"], x["reason"] or "no vectors") for x in excl],
                    cases=[dict(name=x["name"], learner=x["learner"], dir=x["dir"], size=x["size"], f=x["f"],
                                match=x["match"], **x["even"]) for x in used])

    def _print_excl(self, excl):
        if excl:
            self.p(f"    excluded and listed ({len(excl)}): " + "; ".join(f"{x['name']} [{x['reason'] or 'no vectors'}]" for x in excl))

    def _learner_table(self, used):
        if not used:
            return
        self.p("    per learner:  learner       n   median rho   predicted f (range)      min rho    max rho   median cos   median nr")
        for ln in sorted({x["learner"] for x in used}, key=lambda s: (s.split("(")[0], float(re.findall(r"[\d.]+", s)[0]))):
            xs = [x for x in used if x["learner"] == ln]
            rh = [x["even"]["rho"] for x in xs]
            fs = [x["f"] for x in xs]
            fr = f"{min(fs):+.4f}" if max(fs) - min(fs) < 1e-12 else f"{min(fs):+.4f}..{max(fs):+.4f}"
            self.p(f"      {ln:14s} {len(xs):3d}  {_fmt(_med(rh), 11)}   {fr:>20s}  {_fmt(min(rh), 10)} {_fmt(max(rh), 10)}"
                   f"  {_fmt(_med([x['even']['cos'] for x in xs]), 11)} {_fmt(_med([x['even']['nr'] for x in xs]), 11)}")

    def rule_H7(self, group):
        self._need_conv = False
        items = self.pairs(group, "sm", lambda c: c["learner"] in ("DJ(1)", "SL(1)"))
        used = [x for x in items if not x["reason"] and "even" in x]
        excl = [x for x in items if x["reason"] or "even" not in x]
        for x in used:
            x["match"] = x["even"]["nr"] <= CFG["H7_NR"]
        nm = sum(x["match"] for x in used)
        frac = nm / len(used) if used else float("nan")
        verdict = "NO DATA" if not used else ("PASS" if frac >= CFG["RULE_FRAC"] else "FAIL")
        nrs = [x["even"]["nr"] for x in used]
        self.p(f"H7 [literal (finite size)] {group[0]}/{group[1]}: nr <= {CFG['H7_NR']} in {nm}/{len(used)} = {100 * frac:.1f}% [rule >= 90%] -> {verdict}"
               f"   (max nr: DJ(1.0) {_fmt(max([x['even']['nr'] for x in used if x['learner'] == 'DJ(1)'] or [float('nan')]))},"
               f" SL(1)=TH {_fmt(max([x['even']['nr'] for x in used if x['learner'] == 'SL(1)'] or [float('nan')]))})")
        self._print_excl(excl)
        for x in used:
            if not x["match"]:
                self.p(f"      failing: {x['name']} nr={x['even']['nr']:.4g}")
        return dict(rule="H7", label="literal (finite size)", group=group, matched=nm, used=len(used), frac=frac, verdict=verdict,
                    excluded=[(x["name"], x["reason"]) for x in excl],
                    cases=[dict(name=x["name"], learner=x["learner"], nr=x["even"]["nr"], match=x["match"]) for x in used])

    def rule_H6(self, group):
        rs = [r for n, r in self.recs.items() if r["grp"] == "sm" and r["learner"].startswith("PROX")
              and (r["act"], r["start"]) == group]
        planned = [c for c in self.planned.values() if c["grp"] == "sm" and c["learner"].startswith("PROX")
                   and self._group_of(c) == group]
        used = [r for r in rs if not usable(r)]
        excl = [(r["name"], usable(r)) for r in rs if usable(r)] + [(c["name"], "MISSING/FAILED") for c in planned
                                                                      if c["name"] not in self.recs]
        ok = [r for r in used if CFG["H6_RHO"][0] <= r["rho"] <= CFG["H6_RHO"][1] and r["cos"] >= CFG["H6_COS"]]
        frac = len(ok) / len(used) if used else float("nan")
        verdict = "NO DATA" if not used else ("PASS" if frac >= CFG["RULE_FRAC"] else "FAIL")
        self.p(f"H6 {group[0]}/{group[1]} (consistency check of the NP solver): rho in {list(CFG['H6_RHO'])} and cos >= "
               f"{CFG['H6_COS']} in {len(ok)}/{len(used)} = {100 * frac:.1f}% [rule >= 90%] -> {verdict}   "
               f"(rho range {_fmt(min([r['rho'] for r in used] or [float('nan')]))} .. {_fmt(max([r['rho'] for r in used] or [float('nan')]))},"
               f" min cos {_fmt(min([r['cos'] for r in used] or [float('nan')]), 10, 6)})")
        if excl:
            self.p("    excluded and listed: " + "; ".join(f"{a} [{b}]" for a, b in excl))
        for r in used:
            if r not in ok:
                self.p(f"      failing: {r['name']} rho={r['rho']:.5f} cos={r['cos']:.6f}")
        return dict(rule="H6", group=group, matched=len(ok), used=len(used), frac=frac, verdict=verdict, excluded=excl,
                    cases=[dict(name=r["name"], rho=r["rho"], cos=r["cos"], steps=r.get("steps")) for r in used])

    def rule_H5(self, group):
        rs = [r for r in self.recs.values() if r["grp"] == "law" and (r["act"], r["start"]) == group]
        planned = [c for c in self.planned.values() if c["grp"] == "law" and self._group_of(c) == group]
        ok = [r for r in rs if r["law_cos"] >= CFG["H5_COS"] and CFG["H5_RATIO"][0] <= r["law_ratio"] <= CFG["H5_RATIO"][1]]
        frac = len(ok) / len(rs) if rs else float("nan")
        verdict = "NO DATA" if not rs else ("PASS" if frac >= CFG["RULE_FRAC"] else "FAIL")
        self.p(f"H5 {group[0]}/{group[1]}: cos(LAW(C^e), v_L^e) >= {CFG['H5_COS']} and |v_L^e|/|LAW| in {list(CFG['H5_RATIO'])} "
               f"in {len(ok)}/{len(rs)} = {100 * frac:.1f}% [rule >= 90%] -> {verdict}")
        miss = [c["name"] for c in planned if c["name"] not in self.recs]
        if miss:
            self.p("    missing/failed and listed: " + "; ".join(miss))
        self.p("    learner        dir     cos(LAW,v_L^e)  |v_L^e|/|LAW|  kept SV  kept frac   even rho    even cos   even nr  "
               "steps(+/-)   status(+/-)")
        for r in sorted(rs, key=lambda r: (r["dir"], r["learner"])):
            pp = self.recs.get(self.planned[r["name"]]["plus"], {})
            mm = self.recs.get(self.planned[r["name"]]["minus"], {})
            self.p(f"      {r['learner']:12s} {r['dir']:6s} {_fmt(r['law_cos'], 12, 6)} {_fmt(r['law_ratio'], 14, 5)} "
                   f"{r['kept']:8d} {_fmt(r['kept_frac'], 10, 6)} {_fmt(r['even_rho'], 10)} {_fmt(r['even_cos'], 10)} "
                   f"{_fmt(r['even_nr'], 9)}  {pp.get('steps')}/{mm.get('steps')}  {pp.get('learner_status')}/{mm.get('learner_status')}")
        # H5b
        gd = [r for r in self.recs.values() if r["grp"] == "h5" and r["learner"].startswith("GD") and r["sign"] == 1
              and (r["act"], r["start"]) == group and "Clin_rel_err" in r]
        okb = [r for r in gd if r["Clin_rel_err"] <= CFG["H5B_TOL"]]
        fb = len(okb) / len(gd) if gd else float("nan")
        vb = "NO DATA" if not gd else ("PASS" if fb >= CFG["RULE_FRAC"] else "FAIL")
        self.p(f"H5b {group[0]}/{group[1]}: |C+ - C_lin,+|_F/|C+|_F <= {CFG['H5B_TOL']} in {len(okb)}/{len(gd)} = "
               f"{100 * fb:.1f}% [rule >= 90%] -> {vb}   values: "
               + ", ".join(f"{r['learner']} {r['dir']}: {r['Clin_rel_err']:.4f}" for r in sorted(gd, key=lambda r: (r['dir'], r['learner']))))
        return (dict(rule="H5", group=group, matched=len(ok), used=len(rs), frac=frac, verdict=verdict,
                     cases=[{k: r.get(k) for k in ("name", "learner", "dir", "law_cos", "law_ratio", "kept", "kept_frac",
                                                    "even_rho", "even_cos", "even_nr")} for r in rs]),
                dict(rule="H5b", group=group, matched=len(okb), used=len(gd), frac=fb, verdict=vb,
                     cases=[dict(name=r["name"], Clin_rel_err=r["Clin_rel_err"]) for r in gd]))

    def rule_H0(self):
        rs = [r for r in self.recs.values() if r["grp"] == "lin"]
        planned = [c for c in self.planned.values() if c["grp"] == "lin"]
        ok = [r for r in rs if r.get("vL_over_dL", 1) <= CFG["H0_TOL"] and r.get("vNP_over_dNP", 1) <= CFG["H0_TOL"]]
        verdict = "PASS" if rs and len(ok) == len(planned) else "FAIL"
        self.p(f"H0 linear control (196 -> {CFG['N_CLASSES']} linear, init, seed 0): {len(ok)}/{len(planned)} runs with "
               f"|v_L|/|theta_L - theta0| and |v_NP|/|theta_NP - theta0| <= {CFG['H0_TOL']} [rule: all] -> {verdict}   "
               f"(max |v_L|/|dL| = {max([r.get('vL_over_dL', float('nan')) for r in rs] or [float('nan')]):.2e}, "
               f"max |v_NP|/|dNP| = {max([r.get('vNP_over_dNP', float('nan')) for r in rs] or [float('nan')]):.2e}; "
               f"learner status: {sorted(set(r.get('learner_status') for r in rs))})")
        for r in rs:
            if r not in ok:
                self.p(f"      failing: {r['name']} |v_L|/|dL|={r.get('vL_over_dL')} |v_NP|/|dNP|={r.get('vNP_over_dNP')}")
        return dict(rule="H0", group=("linear", "init"), matched=len(ok), used=len(planned), verdict=verdict,
                    cases=[dict(name=r["name"], vL_over_dL=r.get("vL_over_dL"), vNP_over_dNP=r.get("vNP_over_dNP"),
                                status=r.get("learner_status"), steps=r.get("steps")) for r in rs])

    # ---------------- report-only tables ----------------
    def R_tables(self, group):
        out = {}
        g = group
        self.p(f"\n---- R-tables for {g[0]}/{g[1]} (report only, no verdicts) ----")
        # R1
        self.p("R1 ADAM and SGD(0.25,8) on label / rand at s1 (even part; |v_L|/|theta_L - theta0| per sign):")
        self.p("      learner        dir     even rho   even cos    even nr   |vL|/dL(+)  |vL|/dL(-)   steps(+/-)  status(+/-)")
        r1 = []
        for grp, pre in (("r1", "ADAM"), ("h5", "SGD")):
            self._need_conv = False
            for x in self.pairs(g, grp, lambda c: c["learner"].startswith(pre)):
                e = x.get("even", {})
                pp, mm = x["plus"] or {}, x["minus"] or {}
                self.p(f"      {x['learner']:14s} {x['dir']:6s} {_fmt(e.get('rho'), 10)} {_fmt(e.get('cos'), 10)} {_fmt(e.get('nr'), 10)} "
                       f"{_fmt(pp.get('vL_over_dL'), 11)} {_fmt(mm.get('vL_over_dL'), 11)}   {pp.get('steps')}/{mm.get('steps')}"
                       f"  {pp.get('learner_status')}/{mm.get('learner_status')}{'  [' + x['reason'] + ']' if x['reason'] else ''}")
                r1.append(dict(name=x["name"], **e, vLdL_p=pp.get("vL_over_dL"), vLdL_m=mm.get("vL_over_dL")))
        out["R1"] = r1
        # H5 learners report (rho, cos, nr)
        self.p("H5 learners, measured alignment (even part; no rule):")
        h5r = []
        for x in self.pairs(g, "h5", lambda c: True):
            e = x.get("even", {})
            self.p(f"      {x['learner']:14s} {x['dir']:6s} rho {_fmt(e.get('rho'))}  cos {_fmt(e.get('cos'))}  nr {_fmt(e.get('nr'))}"
                   f"{'  [' + x['reason'] + ']' if x['reason'] else ''}")
            h5r.append(dict(name=x["name"], **e))
        out["H5_alignment"] = h5r
        # R2
        self.p("R2 size ladder (label, sign +, single-sign values):")
        self.p("      s/S0   learner         rho        cos         nr      R_lam1      R_trK   NP status    NP its  learner status  steps  TH jump")
        r2 = []
        for c in sorted([c for c in self.planned.values() if c["grp"] == "r2" and self._group_of(c) == g],
                        key=lambda c: (c["frac"], c["learner"])):
            r = self.recs.get(c["name"])
            if r is None:
                self.p(f"      {c['frac']:<6g} {c['learner']:12s} MISSING/FAILED")
                continue
            nr1 = r["nvL"] / r["nvNP"] if r.get("nvNP") else float("nan")
            self.p(f"      {c['frac']:<6g} {c['learner']:12s} {_fmt(r.get('rho'), 10)} {_fmt(r.get('cos'), 10)} {_fmt(nr1, 10)} "
                   f"{_fmt(r.get('R_lam1'), 11)} {_fmt(r.get('R_trK'), 10)}   {str(r.get('np_status')):11s} {str(r.get('np_it')):>6s}  "
                   f"{str(r.get('learner_status')):14s} {r.get('steps')}  {r.get('th_status')}")
            r2.append(dict(frac=c["frac"], learner=c["learner"], rho=r.get("rho"), cos=r.get("cos"), nr=nr1,
                           R_lam1=r.get("R_lam1"), R_trK=r.get("R_trK"), np_status=r.get("np_status"),
                           learner_status=r.get("learner_status"), steps=r.get("steps"), th_status=r.get("th_status")))
        sl = [x for x in r2 if x["learner"] == "SL(32)" and x["frac"] <= 0.25]
        if sl:
            self.p(f"      stated expectation (not a rule): rho of SL(32) in [0.4, 0.6] up to s/S0 = 0.25 -> "
                   f"{'holds' if all(0.4 <= (x['rho'] if math.isfinite(x['rho'] or float('nan')) else -1) <= 0.6 for x in sl) else 'does NOT hold'}"
                   f" (values {', '.join(_fmt(x['rho'], 7).strip() for x in sl)})")
        out["R2"] = r2
        # R3
        self.p("R3 scalar version (single-sign runs of H1-H4): median |R_g - rho| per learner")
        r3 = []
        rs = [r for r in self.recs.values() if r["grp"] == "sm" and (r["act"], r["start"]) == g and r.get("rho") is not None
              and not usable(r) and r["learner"].split("(")[0] in ("SL", "DJ", "GD", "HB") and r["learner"] != "SL(1)"]
        for ln in sorted({r["learner"] for r in rs}):
            xs = [r for r in rs if r["learner"] == ln]
            a = _med([abs(r["R_lam1"] - r["rho"]) for r in xs])
            b = _med([abs(r["R_trK"] - r["rho"]) for r in xs])
            self.p(f"      {ln:14s} n={len(xs):3d}  median |R_lam1 - rho| = {_fmt(a)}   median |R_trK - rho| = {_fmt(b)}"
                   f"   (median rho {_fmt(_med([r['rho'] for r in xs]))}, R_lam1 {_fmt(_med([r['R_lam1'] for r in xs]))}, "
                   f"R_trK {_fmt(_med([r['R_trK'] for r in xs]))})")
            r3.append(dict(learner=ln, n=len(xs), med_abs_Rlam1_rho=a, med_abs_RtrK_rho=b))
        self.p(f"      all: median |R_lam1 - rho| = {_fmt(_med([abs(r['R_lam1'] - r['rho']) for r in rs]))}, "
               f"median |R_trK - rho| = {_fmt(_med([abs(r['R_trK'] - r['rho']) for r in rs]))}")
        out["R3"] = r3
        # R4 replaced by the A1.4 order check
        out["R4_order_check"] = self.order_check(g)
        # R5
        self.p("R5 distances |theta_L - theta0| / |theta_NP - theta0| per learner (must be >= 1):")
        r5 = []
        allr = [r for r in self.recs.values() if (r["act"], r["start"]) == g and r.get("dNP")]
        for ln in sorted({r["learner"] for r in allr}):
            xs = [r["dL"] / r["dNP"] for r in allr if r["learner"] == ln]
            self.p(f"      {ln:14s} n={len(xs):3d}  min {min(xs):.12f}  median {_med(xs):.12f}  max {max(xs):.6f}"
                   f"{'   <-- BELOW 1 (beyond the 1e-9 slack)' if min(xs) < 1 - CFG['NP_DIST_SLACK'] else ''}")
            r5.append(dict(learner=ln, n=len(xs), min=min(xs), median=_med(xs), max=max(xs)))
        out["R5"] = r5
        return out

    def R6(self):
        self.p("R6 spectrum and conditioning of every start:")
        self.p("      combination          lambda_1     lambda_N      tr K0      cond(K0)   ill   fit its  fit mse    eig indices "
               "(1-based, lambda_i/lambda_1)")
        for cn, info in self.si.items():
            di = info.get("directions", {})
            ei = ", ".join(f"{k}: q{di[k]['index']} ({di[k]['lam_ratio']:.4f})" for k in EIG4 if k in di)
            self.p(f"      {cn:20s} {info['lam1']:.4e}  {info['lamN']:.4e}  {info['trK']:.4e}  {info['cond']:.3e}  "
                   f"{'YES' if info['ill_conditioned'] else 'no ':3s}  {str(info.get('fit_iterations', '-')):>6s}  "
                   f"{info.get('fit_mse', float('nan')):.2e}   {ei}")
        return self.si

    # ---------------- AMENDMENT A1: extrapolated rules (SPEC8 sec. 10) ----------------
    def units(self, group, learners, dirs=None, need_conv=False):
        """(combo, learner, dir) -> {"s1": even pair at s1, "s2": even pair at s2} for the small-move group."""
        self._need_conv = need_conv
        sel = lambda c: c["learner"] in learners and (dirs is None or c["dir"] in dirs)
        out = {}
        for x in self.pairs(group, "sm", sel):
            if x["size"] in ("s1", "s2"):
                out.setdefault((x["combo"], x["learner"], x["dir"]), {})[x["size"]] = x
        self._need_conv = False
        return out

    @staticmethod
    def _unit_reason(u, sizes=("s1", "s2")):
        for sz in sizes:
            x = u.get(sz)
            if x is None:
                return f"{sz}: NOT PLANNED"
            if x["reason"]:
                return f"{sz}: {x['reason']}"
            if "even" not in x:
                return f"{sz}: no vectors"
        return ""

    def rule_x(self, hname, group, learners, tol, dirs=None, need_conv=False, conv_frac=None):
        """A1.2/A1.3: rho_x = (4 rho(s2) - rho(s1))/3, cos_x = cos(s2), nr_x = nr(s2); same matches() and tol."""
        U = self.units(group, learners, dirs, need_conv)
        used, excl = [], []
        for k, u in sorted(U.items()):
            why = self._unit_reason(u)
            if why:
                excl.append((k, why))
                continue
            e1, e2 = u["s1"]["even"], u["s2"]["even"]
            f = u["s1"]["f"]
            rx = (4.0 * e2["rho"] - e1["rho"]) / 3.0
            used.append(dict(combo=k[0], learner=k[1], dir=k[2], f=f, rho_s1=e1["rho"], rho_s2=e2["rho"], rho_x=rx,
                             cos_x=e2["cos"], nr_x=e2["nr"], cos_s1=e1["cos"],
                             match=matches(rx, e2["cos"], e2["nr"], f, tol)))
        nm = sum(x["match"] for x in used)
        frac = nm / len(used) if used else float("nan")
        verdict = "NO DATA" if not used else ("PASS" if frac >= CFG["RULE_FRAC"] else "FAIL")
        extra = ""
        if conv_frac is not None:
            def conv(u):
                return all(u.get(sz) is not None and u[sz]["plus"] is not None and u[sz]["minus"] is not None
                           and u[sz]["plus"].get("learner_status") == "CONVERGED"
                           and u[sz]["minus"].get("learner_status") == "CONVERGED" for sz in ("s1", "s2"))
            nconv = sum(conv(u) for u in U.values())
            cf = nconv / len(U) if U else float("nan")
            extra = (f"; converged (both sizes, both signs) {nconv}/{len(U)} planned = {100 * cf:.0f}% "
                     f"[rule >= {100 * conv_frac:.0f}%]")
            if used and cf < conv_frac:
                verdict = "FAIL"
        self.p(f"{hname} [PRIMARY, extrapolated] {group[0]}/{group[1]}: matched {nm}/{len(used)} (learner, direction) "
               f"pairs = {100 * frac:.1f}% [rule >= {100 * CFG['RULE_FRAC']:.0f}%, tol {tol}]{extra} -> {verdict}")
        if excl:
            self.p(f"    excluded and listed ({len(excl)}): " + "; ".join(f"{k[0]} {k[1]} {k[2]} [{w}]" for k, w in excl))
        self.p("    combination        learner          dir      rho(s1)    rho(s2)      rho_x          f   cos(s2)    nr(s2)  match")
        for x in used:
            self.p(f"      {x['combo']:16s} {x['learner']:15s} {x['dir']:5s} {_fmt(x['rho_s1'], 10)} {_fmt(x['rho_s2'], 10)} "
                   f"{_fmt(x['rho_x'], 10)} {_fmt(x['f'], 10)} {_fmt(x['cos_x'], 9)} {_fmt(x['nr_x'], 9)}  "
                   f"{'yes' if x['match'] else 'NO'}")
        return dict(rule=hname, label="PRIMARY", group=group, matched=nm, used=len(used), frac=frac, verdict=verdict,
                    excluded=[(" ".join(k), w) for k, w in excl], cases=used)

    def rule_H7x(self, group):
        U = self.units(group, ["DJ(1)", "SL(1)"])
        used, excl = [], []
        for k, u in sorted(U.items()):
            why = self._unit_reason(u, sizes=("s2",))
            if why:
                excl.append((k, why))
                continue
            nr2 = u["s2"]["even"]["nr"]
            used.append(dict(combo=k[0], learner=k[1], dir=k[2], nr_s2=nr2, match=nr2 <= CFG["H7_NR"]))
        nm = sum(x["match"] for x in used)
        frac = nm / len(used) if used else float("nan")
        verdict = "NO DATA" if not used else ("PASS" if frac >= CFG["RULE_FRAC"] else "FAIL")
        mx = lambda ln: max([x["nr_s2"] for x in used if x["learner"] == ln] or [float("nan")])
        self.p(f"H7x [PRIMARY] {group[0]}/{group[1]}: nr(s2) <= {CFG['H7_NR']} in {nm}/{len(used)} (learner, direction) pairs "
               f"= {100 * frac:.1f}% [rule >= 90%] -> {verdict}   (max nr(s2): DJ(1.0) {_fmt(mx('DJ(1)'))}, SL(1)=TH {_fmt(mx('SL(1)'))})")
        if excl:
            self.p(f"    excluded and listed ({len(excl)}): " + "; ".join(f"{k[0]} {k[1]} {k[2]} [{w}]" for k, w in excl))
        for x in used:
            if not x["match"]:
                self.p(f"      failing: {x['combo']} {x['learner']} {x['dir']} nr(s2)={x['nr_s2']:.4g}")
        return dict(rule="H7x", label="PRIMARY", group=group, matched=nm, used=len(used), frac=frac, verdict=verdict,
                    excluded=[(" ".join(k), w) for k, w in excl], cases=used)

    def order_check(self, g):
        """A1.4: |rho(s1) - f|, |rho(s2) - f|, ratio (expected ~4), rho_x for every (learner, direction)."""
        lo, hi = CFG["ORDER_RATIO_RANGE"]
        self.p(f"R4 (A1.4 order check, replaces the old R4; even part): ratio |rho(s1)-f|/|rho(s2)-f| expected ~4; "
               f"FLAG if outside [{lo:g}, {hi:g}] while |rho(s1)-f| > {CFG['ORDER_MIN_ERR']:g}")
        self.p("      combination      learner          dir    |rho(s1)-f|  |rho(s2)-f|    ratio       rho_x          f   note")
        lns = ([f"SL({m})" for m in CFG["SL_MS"] if m > 1] + [f"DJ({c:g})" for c in CFG["DJ_CS"]]
               + [f"GD({c:g})" for c in CFG["GD_CS"]] + [f"HB({c:g},{CFG['HB_MU']:g})" for c in CFG["HB_CS"]])
        U = self.units(g, lns)
        order = {ln: i for i, ln in enumerate(lns)}
        rows, nflag = [], 0
        for k in sorted(U, key=lambda k: (k[0], order[k[1]], DIRS6.index(k[2]))):
            u = U[k]
            why = self._unit_reason(u)
            if why:
                self.p(f"      {k[0]:16s} {k[1]:15s} {k[2]:5s}  [{why}]")
                rows.append(dict(combo=k[0], learner=k[1], dir=k[2], note=why))
                continue
            f = u["s1"]["f"]
            r1, r2 = u["s1"]["even"]["rho"], u["s2"]["even"]["rho"]
            a, b = abs(r1 - f), abs(r2 - f)
            ratio = a / b if b > 0 else float("inf")
            rx = (4.0 * r2 - r1) / 3.0
            flag = a > CFG["ORDER_MIN_ERR"] and not (lo <= ratio <= hi)
            nflag += flag
            self.p(f"      {k[0]:16s} {k[1]:15s} {k[2]:5s} {_fmt(a, 12, 6)} {_fmt(b, 12, 6)} {_fmt(ratio, 8, 2)} "
                   f"{_fmt(rx, 11)} {_fmt(f, 10)}   {'FLAG' if flag else ''}")
            rows.append(dict(combo=k[0], learner=k[1], dir=k[2], err_s1=a, err_s2=b, ratio=ratio, rho_x=rx, f=f, flag=flag))
        self.p(f"      flagged: {nflag} of {sum('ratio' in r for r in rows)}")
        return rows

    def run(self):
        groups = sorted({self._group_of(c) for c in self.planned.values() if c["grp"] != "lin"})
        H = []
        self.p("=" * 110)
        self.p(f"H-TABLE (pre-registered rules; data_source={DATA_SOURCE}; MODE={CFG['MODE']}; "
               f"{len(self.recs)} case records, {len(self.failed)} failed, {len(self.missing)} missing)")
        self.p("=" * 110)
        if self.failed:
            self.p("!!! FAILED CASES (traceback in part8_run/failed/):")
            for f in self.failed:
                self.p(f"!!!   {f['name']}: {f['error'].strip().splitlines()[-1]}")
        if self.missing:
            self.p(f"!!! {len(self.missing)} planned cases have no result yet (run interrupted?): "
                   + ", ".join(self.missing[:20]) + (" ..." if len(self.missing) > 20 else ""))
        H.append(self.rule_H0())
        for g in groups:
            arm = "robustness (non-smooth activation)" if g[0] == "relu" else "primary arm"
            self.p(f"\n---- {g[0]}/{g[1]} ({arm}) ----")
            sl = [f"SL({m})" for m in CFG["SL_MS"] if m > 1]
            dj = [f"DJ({c:g})" for c in CFG["DJ_CS"]]
            gd = [f"GD({c:g})" for c in CFG["GD_CS"]]
            hb = [f"HB({c:g},{CFG['HB_MU']:g})" for c in CFG["HB_CS"]]
            cf = CFG["H3_CONV_FRAC"]
            lit = "literal (finite size)"
            self.p("-- PRIMARY: extrapolated rules (AMENDMENT A1, SPEC8 sec. 10) and H5, H5b, H6 --")
            H.append(self.rule_x("H1x", g, sl, CFG["TOL"]["H1"]))
            H.append(self.rule_x("H2x", g, dj, CFG["TOL"]["H2"]))
            H.append(self.rule_x("H3x", g, gd, CFG["TOL"]["H3"], dirs=EIG4, need_conv=True, conv_frac=cf))
            H.append(self.rule_x("H4x", g, hb, CFG["TOL"]["H4"], dirs=EIG4, need_conv=True, conv_frac=cf))
            H.extend(self.rule_H5(g))
            H.append(self.rule_H6(g))
            H.append(self.rule_H7x(g))
            self.p("-- literal (finite size): the rules as first written, each size counted separately --")
            H.append(self.rule_match("H1", g, sl, CFG["TOL"]["H1"], label=lit))
            H.append(self.rule_match("H2", g, dj, CFG["TOL"]["H2"], label=lit))
            H.append(self.rule_match("H3", g, gd, CFG["TOL"]["H3"], dirs=EIG4, need_conv=True, conv_frac=cf, label=lit))
            H.append(self.rule_match("H4", g, hb, CFG["TOL"]["H4"], dirs=EIG4, need_conv=True, conv_frac=cf, label=lit))
            H.append(self.rule_H7(g))
        R = {f"{g[0]}/{g[1]}": self.R_tables(g) for g in groups}
        R["R6"] = self.R6()
        self.p("\n" + "=" * 110 + "\nVERDICT SUMMARY (data_source=" + str(DATA_SOURCE) + ")")
        for h in H:
            h.setdefault("label", "PRIMARY")
        for lab in ("PRIMARY", "literal (finite size)"):
            self.p(f" {lab}:")
            for h in [h for h in H if h["label"] == lab]:
                fr = h.get("frac")
                pct = f" = {100 * fr:.1f}%" if isinstance(fr, float) and math.isfinite(fr) else ""
                self.p(f"  {h['rule']:4s} {h['group'][0]}/{h['group'][1]:8s} {h['verdict']:8s} ({h['matched']}/{h['used']}{pct})")
        return H, R


def all_case_scalars(recs):
    return {n: dict(r) for n, r in recs.items()}

# =============================================================================================================
# Environment check, timing estimate, saving
# =============================================================================================================
def env_check():
    """Inspect and print only (never installs anything)."""
    say(f"python {platform.python_version()} | torch {torch.__version__} | torch.version.cuda = {torch.version.cuda} | "
        f"numpy {np.__version__} | platform {platform.platform()}")
    say(f"libraries: torchvision={'yes' if HAVE_TORCHVISION else 'NO (MNIST unavailable -> synthetic fallback)'} | "
        f"scipy={'yes' if HAVE_SCIPY else 'no (not needed)'}")
    ng = torch.cuda.device_count() if torch.cuda.is_available() else 0
    say(f"CUDA available: {torch.cuda.is_available()} | GPUs: {ng}")
    for i in range(ng):
        pr = torch.cuda.get_device_properties(i)
        say(f"  GPU {i}: {pr.name}, {pr.total_memory / 1024 ** 3:.1f} GB")
    dev = torch.device("cuda" if ng >= 1 else "cpu")
    # float64 smoke test on the compute device
    a = torch.randn(64, 64, dtype=DT, device=dev, generator=None)
    K = a @ a.T + torch.eye(64, dtype=DT, device=dev)
    lam = torch.linalg.eigvalsh(K)
    Lc = torch.linalg.cholesky(K)
    x = torch.cholesky_solve(torch.ones(64, 1, dtype=DT, device=dev), Lc)
    res = float((K @ x - 1).abs().max())
    X = torch.randn(5, 196, dtype=DT, device=dev)
    prob, th = layered_problem(X, [196, 16, 16, 10], "tanh", 0)
    th = th.to(dev)
    J = prob.J(th)
    ok = (J.dtype == torch.float64 and th.dtype == torch.float64 and K.dtype == torch.float64 and res < 1e-10
          and bool(torch.isfinite(lam).all()) and J.shape == (50, th.numel()))
    say(f"float64 smoke test on {dev} (matmul, eigh, cholesky, jacrev): {'OK' if ok else 'FAILED'} "
        f"(default dtype {torch.get_default_dtype()}, solve residual {res:.1e})")
    if not ok:
        raise RuntimeError("float64 smoke test failed")
    if ng == 0:
        say("\n" + "#" * 96 + "\n#### WARNING: NO GPU VISIBLE. Everything runs on the CPU (correct, but only MODE='tiny' is practical).\n"
            "#### On the GPU machine check that this PyTorch build sees the GPUs (torch.cuda.is_available()).\n" + "#" * 96)
    set_device(dev)
    say(f"==> computations run on: {DEVICE} | torch CPU threads: {torch.get_num_threads()}")


def _time(fn, reps=3):
    ts = []
    for _ in range(reps):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t1)
    return float(np.median(ts))


def timing_estimate(plan, run_dir, n_gpus=1):
    """Times one Jacobian, one NP solve, one LAW term and one first-order step; prints rough totals."""
    items = [it for it in plan if not it.get("kernel_only")]
    if not items:
        say("no combinations to estimate")
        return {}
    ctx = Ctx(items[0]["combo"], run_dir)
    prob, st = ctx.prob, ctx.st
    s1 = CFG["S1_FRAC"] * ctx.S0
    tJ = _time(lambda: gram(prob.J(st.th0)))
    tNP = _time(lambda: nearest_point(prob, st.th0, st.F0 + s1 * ctx.dirs["label"], st.th0), reps=1)
    d = ctx.dirs["label"]
    tLAW1 = _time(lambda: law(prob, st, s1 * s1 * torch.outer(d, d)), reps=1)
    L = L_GD(0.25, 0.0, 200)
    tstep = _time(lambda: run_first_order(prob, st, st.F0 + s1 * d, s1, L), reps=1) / 200
    nterms = min(prob.N, CFG["LAW_MAX_TERMS"])
    say(f"timing on {DEVICE} ({ctx.cname}, N={prob.N}, P={prob.P}, cond(K0) = {st.lam1 / st.lamN:.3e}): one Jacobian+K {tJ * 1e3:.1f} ms | one NP solve "
        f"{tNP:.2f} s | one LAW term {tLAW1:.2f} s (a full-rank C uses up to {nterms} terms: {fmt_time(nterms * tLAW1)}) | "
        f"one first-order step {tstep * 1e3:.2f} ms")
    lr = [float(x) for x in (st.lam / st.lam1).cpu()]
    lamN = lr[-1]
    meas = 7                                                 # Jacobians per measurement (NP ~4, TH, g at L/NP/TH)

    def steps_first_order(c):
        cs = c["L"]
        rat = ctx.dinfo[c["dir"]]["lam_ratio"]
        rat = lamN if rat is None else rat
        ci = cs["c"] * rat if cs["kind"] in ("GD", "HB", "SGD") else 1e-3
        tol = cs["stop_rel"]
        if cs["kind"] == "HB":
            rate = min(ci / (1 - cs["mu"]), 1 - math.sqrt(cs["mu"]))
        elif cs["kind"] == "ADAM":
            rate = 1e-3
        else:
            rate = min(ci, 1.0)
        return min(cs["cap"], math.log(1 / max(tol, 1e-12)) / max(rate, 1e-12))

    lo = hi = 0.0
    for c in items[0]["cases"]:
        k = c["L"]["kind"]
        if c["kind"] == "law":
            v = 2 * nterms * tJ
            lo += v
            hi += v
            continue
        base = meas * tJ
        if k == "SL":
            v = base + c["L"]["m"] * tJ
            lo += v
            hi += v
        elif k == "DJ":
            cc = c["L"]["c"]
            n = 4 if cc == 1.0 else math.log(1e-7) / math.log(abs(1 - cc))
            lo += base + n * tJ
            hi += base + n * tJ
        elif k == "PROX":
            lo += base + 6 * tJ
            hi += base + 6 * tJ
        else:
            lo += base + steps_first_order(c) * tstep
            hi += base + c["L"]["cap"] * tstep
    nc = len(items)
    say(f"estimated grid time per combination: {fmt_time(lo)} (first-order learners converge at their linear rate) to "
        f"{fmt_time(hi)} (every first-order run hits its cap)")
    say(f"this MODE ({CFG['MODE']}): {nc} combination(s) -> about {fmt_time(lo * nc)} .. {fmt_time(hi * nc)} on one device; "
        f"with {max(1, n_gpus)} GPU worker(s) (static split by combination): {fmt_time(lo * math.ceil(nc / max(1, n_gpus)))}"
        f" .. {fmt_time(hi * math.ceil(nc / max(1, n_gpus)))}")
    if CFG["MODE"] != "tiny":
        say(f"MODE='full' (18 combinations of this size): about {fmt_time(18 * lo)} .. {fmt_time(18 * hi)} GPU time"
            f"{' -- MORE THAN 8 GPU-HOURS (grid NOT changed, as the spec requires)' if 18 * lo > 8 * 3600 else ''}")
    del ctx
    free_memory()
    return dict(t_J=tJ, t_NP=tNP, t_LAW_term=tLAW1, t_step=tstep, per_combo_lo=lo, per_combo_hi=hi, combos=nc)


def save_outputs(H, R, recs, failed, missing, test_results, start_infos, timing, analysis_lines):
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               device=str(DEVICE), gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
               if torch.cuda.is_available() else [], platform=platform.platform(), threads=torch.get_num_threads())
    res = dict(spec="SPEC8", mode=CFG["MODE"], data_source=DATA_SOURCE, signature=config_signature(),
               written=time.strftime("%Y-%m-%d %H:%M:%S"), config={k: v for k, v in CFG.items()}, environment=env,
               tests=test_results, starts=start_infos, timing=timing, H_table=H, R_tables=R,
               failed=[dict(name=f["name"], error=f["error"]) for f in failed], missing=missing,
               cases=recs, analysis_text="\n".join(analysis_lines))
    write_json_atomic(CFG["RESULTS_FILE"], res)
    rd = CFG["RUN_DIR"]
    with zipfile.ZipFile(CFG["ZIP_FILE"], "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CFG["RESULTS_FILE"])
        for sub in ("done", "failed", "starts", "logs"):
            for p in sorted(glob.glob(os.path.join(rd, sub, "*"))):
                if sub == "starts" and p.endswith(".pt"):
                    continue
                z.write(p)
    say(f"wrote {os.path.abspath(CFG['RESULTS_FILE'])} ({os.path.getsize(CFG['RESULTS_FILE']) / 1e6:.2f} MB)")
    say(f"wrote {os.path.abspath(CFG['ZIP_FILE'])} ({os.path.getsize(CFG['ZIP_FILE']) / 1e6:.2f} MB; the vecs folder "
        f"{os.path.abspath(os.path.join(rd, 'vecs'))} is NOT in the zip)")
    return res
