"""part9lib - library of evaluation_point_tests.ipynb (SPEC9: three follow-up tests of the evaluation-point law).
Written by the notebook's %%writefile cells; imported by the notebook kernel and by every worker process.
Edit the notebook cells, not this file. The functions listed in SPEC9 sec. 2 are VERBATIM copies from the reviewed
alignment_fraction_test.ipynb (md5 787d12332b4bf0eb4f862ecbfc06d571); every building block takes a Problem
(callables F(theta), J(theta)), so the SAME code runs on the solvable 2-parameter model, the small test networks,
the linear control and the MLP."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (must precede CUDA init)
import sys, re, json, time, math, hashlib, platform, traceback, subprocess, glob, gc, zipfile, shutil, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.func import jacrev

torch.set_default_dtype(torch.float64)            # float64 EVERYWHERE
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
LIBNAME = "part9lib"


# ---- VERBATIM from alignment_fraction_test.ipynb (SPEC9 sec. 2; checked character by character) ----
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


# ---- NEW in SPEC9 ----
_HB = {"t": time.time(), "label": ""}


def progress(msg):
    """NEW (SPEC9 10.5): print a progress line and append it to <RUN_DIR>/progress.log."""
    say(msg)
    rd = CFG.get("RUN_DIR")
    if rd:
        try:
            os.makedirs(rd, exist_ok=True)
            with open(os.path.join(rd, "progress.log"), "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def heartbeat(msg="", force=False):
    """Progress line at least every PROGRESS_EVERY_S seconds (called from inside long loops); also to progress.log."""
    now = time.time()
    if force or now - _HB["t"] >= CFG.get("PROGRESS_EVERY_S", 60):
        _HB["t"] = now
        progress(f"  [{time.strftime('%H:%M:%S')}] {_HB['label']} {msg}")


def test_fail_hook(name):
    """TEST HOOK, inert unless the environment variable PART9_TEST_FAIL_KEY is set: a case whose name contains
    that string raises on purpose (used only to check the failure listing)."""
    key = os.environ.get("PART9_TEST_FAIL_KEY", "")
    if key and key in name:
        raise RuntimeError(f"deliberate failure injected by PART9_TEST_FAIL_KEY={key!r} (test hook)")


# =============================================================================================================
# Generic problem, networks, linear algebra, start
# =============================================================================================================
# ---- VERBATIM from alignment_fraction_test.ipynb (SPEC9 sec. 2; checked character by character) ----
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
# SPEC8 building blocks: exact jump, SPEC8 learners, nearest point, measurement, even part, LAW (unit tests)
# =============================================================================================================
# ---- VERBATIM from alignment_fraction_test.ipynb (SPEC9 sec. 2; checked character by character) ----
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
# Data, directions, files, learner constructors, matches
# =============================================================================================================
# ---- VERBATIM from alignment_fraction_test.ipynb (SPEC9 sec. 2; checked character by character) ----
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


def linear_combo():
    return dict(act="linear", start="init", seed=0)


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



# =============================================================================================================
# Reviewed unit tests U1-U14 (unchanged)
# =============================================================================================================
# ---- VERBATIM from alignment_fraction_test.ipynb (SPEC9 sec. 2; checked character by character) ----
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



# =============================================================================================================
# SPEC9 sec. 2 allowed additions: jump_a (copy of jump that also returns a), measure_ext (measure + theta_NP, theta_TH)
# =============================================================================================================
def jump_a(prob, thb, t, Jb=None):
    """NEW (SPEC9 sec. 2): a COPY of jump() - same algorithm, same arithmetic, same statuses - that also returns the
    coefficient vector a (theta = theta_b + J_b^T a). jump() itself is untouched."""
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
    return dict(theta=th, status=status, it_chord=it_chord, it_newton=it_newton, res_inf=rinf, a=a)


def measure_ext(prob, st, thL, np_cap=None):
    """NEW (SPEC9 sec. 2): the same steps as measure() (copied line by line) and additionally returns theta_NP and
    theta_TH of the SAME fibre (needed for Part G). Returns (scalars, v_L, v_NP, theta_NP, theta_TH)."""
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
    return sc, vL, vNP, thNP, thTH


# =============================================================================================================
# SPEC9 sec. 3: new learners
# =============================================================================================================
AJ_BAD = ("AJ-FAILED", "AJ-INCONSISTENT")


def alpha_jump(prob, thb, t, alpha, log=None, keep_points=False):
    """NEW. SPEC9 3.1 alpha-point jump AJ(theta_b, t; alpha): theta with F(theta) = t and theta - theta_b = J(theta_e)^T a,
    theta_e = theta_b + alpha (theta - theta_b). Outer fixed point on theta; inner exact solve for a (= jump_a with base
    theta_b and the matrix J_e: chord with K_e = J_e J_e^T from a = 0 to |F - t|_inf <= JUMP_TOL, cap 200, then Newton with
    M = J(theta_b + J_e^T a) J_e^T, cap 50). Damping (half step) when the outer step length did not decrease.
    Statuses: CONVERGED | AJ-FAILED (inner failure or outer cap) | AJ-INCONSISTENT (hz > AJ_HZ_TOL).
    log: list to append dict(Fe, b[, the]) with theta_e the FINAL evaluation point and b = J(theta_e)(theta - theta_b)."""
    th = thb.clone()
    prev_step, n_damp, it, status, why = None, 0, 0, None, ""
    the_c, Je = None, None                      # J is a deterministic function of theta_e: recomputed only if theta_e changed
    ic = inw = 0
    while True:
        if it >= CFG["AJ_OUTER_CAP"]:
            status, why = "AJ-FAILED", "outer cap"
            break
        the = thb + alpha * (th - thb)
        if the_c is None or not torch.equal(the, the_c):
            Je = None
            Je = prob.J(the)
            the_c = the
        jr = jump_a(prob, thb, t, Jb=Je)
        it += 1
        ic += jr["it_chord"]
        inw += jr["it_newton"]
        if jr["status"] != "CONVERGED":
            status, why = "AJ-FAILED", "inner solve failed"
            break
        new = jr["theta"]
        step = vnorm(new - th)
        if step <= CFG["AJ_OUT_TOL_REL"] * vnorm(new - thb) + CFG["AJ_OUT_TOL_ABS"]:
            th = new
            status = "CONVERGED"
            step_vec = Je.T @ jr["a"]           # the displacement theta - theta_b as computed (before rounding into theta)
            break
        if prev_step is not None and not (step < prev_step):
            th = th + 0.5 * (new - th)          # damping (counted)
            n_damp += 1
        else:
            th = new
        prev_step = step
        heartbeat(f"AJ({alpha:g}) outer it {it} step {step:.2e}")
    out = dict(theta=th, status=status, status_supp=status, outer_it=it, dampings=n_damp, inner_chord=ic, inner_newton=inw,
               why=why, res_inf=float("nan"), hz=float("nan"), hz_bound=float("nan"), hz_step=float("nan"))
    if status != "CONVERGED":
        return out
    the = thb + alpha * (th - thb)               # final evaluation point (for the FINAL theta)
    Jf = Je if torch.equal(the, the_c) else prob.J(the)
    cons = aj_consistency(prob, thb, th, alpha, Jf=Jf)      # A3.1: hz from the stored vectors, rounding bound B
    out["hz"], out["hz_bound"] = cons["hz"], cons["hz_bound"]
    Lf = chol(gram(Jf))
    Jd = Jf @ (th - thb)                         # for the log: b = J(theta_e)(theta - theta_b)
    # A3.1: reported number - the same quantity evaluated on the step vector J_e^T a (equal to theta - theta_b in exact
    # arithmetic, free of the cancellation error ~ eps |theta| / |theta - theta_b| of the subtraction)
    ns = vnorm(step_vec)
    vz2 = step_vec - Jf.T @ csolve(Lf, Jf @ step_vec)
    out["hz_step"] = vnorm(vz2) / ns if ns > 0 else 0.0
    Je = Jf = None
    out["res_inf"] = float((prob.F(th) - t).abs().max())
    if cons["inconsistent"]:
        out["status"] = "AJ-INCONSISTENT"
    if not (out["hz_step"] <= CFG["AJ_HZ_TOL"]):
        out["status_supp"] = "AJ-INCONSISTENT"
    if log is not None:
        item = dict(Fe=prob.F(the).clone(), b=Jd.clone())   # b = J(theta_e)(theta - theta_b) (= K(theta_e) a)
        if keep_points:
            item["the"] = the.clone()
        log.append(item)
    return out


FP_EPS = float(torch.finfo(DT).eps)               # 2.220446049250313e-16


def aj_consistency(prob, thb, th, alpha, Jf=None):
    """NEW (A3.1). Consistency of an alpha-point jump theta_b -> theta: hz = |V_e(theta - theta_b)|/|theta - theta_b| with V_e the
    vertical projector at theta_e = theta_b + alpha (theta - theta_b), computed from the STORED vectors (arithmetic as delivered);
    rounding bound B = eps |theta| / |theta - theta_b|; inconsistent <=> hz > AJ_HZ_TOL + AJ_HZ_FP * B. |theta - theta_b| = 0:
    hz = B = 0. Jf: the Jacobian at theta_e if the caller has it."""
    dth = th - thb
    nd = vnorm(dth)
    if nd == 0:
        return dict(hz=0.0, hz_bound=0.0, inconsistent=False)
    if Jf is None:
        Jf = prob.J(thb + alpha * (th - thb))
    Lf = chol(gram(Jf))
    Jd = Jf @ dth
    vz = dth - Jf.T @ csolve(Lf, Jd)
    hz = vnorm(vz) / nd
    B = FP_EPS * vnorm(th) / nd
    return dict(hz=hz, hz_bound=B, inconsistent=bool(not (hz <= CFG["AJ_HZ_TOL"] + CFG["AJ_HZ_FP"] * B)))


def _aj_acc(acc, r):
    if math.isfinite(r.get("hz", float("nan"))):
        acc["hz_excess_max"] = max(acc["hz_excess_max"], r["hz"] - CFG["AJ_HZ_FP"] * r["hz_bound"])
        acc["lit_flags"] += int(r["hz"] > 1e-7)
    acc["outer_it"].append(r["outer_it"])
    acc["dampings"] += r["dampings"]
    acc["hz_max"] = max(acc["hz_max"], r["hz"]) if math.isfinite(r["hz"]) else acc["hz_max"]
    acc["hz_step_max"] = max(acc["hz_step_max"], r["hz_step"]) if math.isfinite(r["hz_step"]) else acc["hz_step_max"]
    acc["res_max"] = max(acc["res_max"], r["res_inf"]) if math.isfinite(r["res_inf"]) else acc["res_max"]
    acc["n_incons"] += int(r["status"] == "AJ-INCONSISTENT")
    acc["n_incons_supp"] += int(r["status_supp"] == "AJ-INCONSISTENT")


def _aj_new():
    return dict(outer_it=[], dampings=0, hz_max=0.0, hz_step_max=0.0, res_max=0.0, n_incons=0, n_incons_supp=0,
                hz_excess_max=float("-inf"), lit_flags=0)


def _aj_out(acc):
    o = acc["outer_it"]
    return dict(aj_calls=len(o), aj_outer_total=int(sum(o)), aj_outer_max=int(max(o)) if o else 0,
                aj_dampings=int(acc["dampings"]), aj_hz_max=acc["hz_max"], aj_hz_step_max=acc["hz_step_max"],
                aj_res_max=acc["res_max"], aj_inconsistent=acc["n_incons"], aj_inconsistent_supp=acc["n_incons_supp"],
                aj_hz_excess_max=acc["hz_excess_max"] if math.isfinite(acc["hz_excess_max"]) else None,
                aj_lit_flags=int(acc["lit_flags"]))


def _aj_status(base, acc, failed):
    """Run status: AJ-FAILED if a sub-jump failed; else AJ-INCONSISTENT if any sub-jump was inconsistent (A3.1: hz > AJ_HZ_TOL + AJ_HZ_FP B);
    else the run's own status. status_supp: the same with hz evaluated on the step vector (SUPPLEMENTARY)."""
    if failed:
        return "AJ-FAILED", "AJ-FAILED"
    lit = "AJ-INCONSISTENT" if acc["n_incons"] else base
    sup = "AJ-INCONSISTENT" if acc["n_incons_supp"] else base
    return lit, sup


def run_AJ(prob, st, t, alpha, log=None, keep_points=False):
    """NEW. Learner AJ(alpha) := AJ(theta0, t; alpha). Prediction f = alpha."""
    r = alpha_jump(prob, st.th0, t, alpha, log=log, keep_points=keep_points)
    acc = _aj_new()
    _aj_acc(acc, r)
    out = dict(theta=r["theta"], status=r["status"], status_supp=r["status_supp"], steps=1, aj_why=r["why"])
    out.update(_aj_out(acc))
    return out


def run_SLa(prob, st, t, m, alpha, log=None, keep_points=False):
    """NEW. SPEC9 3.2 SLalpha(m, alpha): theta <- AJ(theta, F0 + (k/m)(t - F0); alpha), k = 1..m.
    Prediction f = 1/2 + (alpha - 1/2)/m. Stops at the first FAILED sub-jump; an AJ-INCONSISTENT sub-jump is counted
    (the run is then AJ-INCONSISTENT, excluded and listed) and the transport continues."""
    th = st.th0.clone()
    acc, why, k, failed = _aj_new(), "", 0, False
    for k in range(1, m + 1):
        tk = st.F0 + (k / m) * (t - st.F0)
        r = alpha_jump(prob, th, tk, alpha, log=log, keep_points=keep_points)
        _aj_acc(acc, r)
        th = r["theta"]
        if r["status"] != "CONVERGED" and not why:
            why = f"sub-jump {k}: {r['why'] or r['status']}"
        if r["status"] == "AJ-FAILED":
            failed = True
            break
    status, status_supp = _aj_status("CONVERGED", acc, failed)
    out = dict(theta=th, status=status, status_supp=status_supp, steps=k, aj_why=why)
    out.update(_aj_out(acc))
    return out


def run_TJ(prob, st, t, c, alpha, log=None, keep_points=False):
    """NEW. SPEC9 3.3 theta-method in output space TJ(c, alpha): q = (1 - c(1 - alpha))/(1 + c alpha);
    repeat r = F - t; stop at |r|_2 <= TJ_TOL_REL |t - F0|_2 (cap TJ_CAP); theta <- AJ(theta, t + q r; alpha).
    Prediction f = (1 + kappa)/(2 + kappa), kappa = c(2 alpha - 1)."""
    q = (1.0 - c * (1.0 - alpha)) / (1.0 + c * alpha)
    th = st.th0.clone()
    target = CFG["TJ_TOL_REL"] * vnorm(t - st.F0)
    acc, status, why, k, failed = _aj_new(), None, "", 0, False
    while True:
        r = prob.F(th) - t
        rn = vnorm(r)
        if rn <= target:
            status = "CONVERGED"
            break
        if not math.isfinite(rn):
            status = "DIVERGED"
            break
        if k >= CFG["TJ_CAP"]:
            status = "UNCONVERGED"
            break
        aj = alpha_jump(prob, th, t + q * r, alpha, log=log, keep_points=keep_points)
        _aj_acc(acc, aj)
        th = aj["theta"]
        k += 1
        if aj["status"] != "CONVERGED" and not why:
            why = f"step {k}: {aj['why'] or aj['status']}"
        if aj["status"] == "AJ-FAILED":
            failed = True
            break
        heartbeat(f"TJ({c:g},{alpha:g}) step {k} |F-t|={rn:.3e}")
    status, status_supp = (status, status) if status == "DIVERGED" else _aj_status(status, acc, failed)
    out = dict(theta=th, status=status, status_supp=status_supp, steps=k, final_res=rn, q=q, aj_why=why)
    out.update(_aj_out(acc))
    return out


def run_TM(prob, st, t, s, c, alpha):
    """NEW. SPEC9 3.4 theta-method in weight space TM(c, alpha): eta = c N / lambda_1(K0);
    theta+ = theta - eta grad L(theta + alpha (theta+ - theta); t). delta = theta+ - theta by the quasi-Newton iteration
    delta <- delta - (I + a J_e^T J_e)^{-1} g(delta), g = delta + (eta/N) J_e^T (F(theta_e) - t), theta_e = theta + alpha delta,
    a = eta alpha / N, (I + a J^T J)^{-1} g = g - J^T (J J^T + I/a)^{-1} J g; start delta = 0; stop (A3.2) when (i) |g| <= 1e-12 |delta| + 1e-16
    or (ii) |g| <= TM_INNER_FP eps (1 + c) |theta| (theta at the start of the step; rounding floor of g);
    cap TM_INNER_CAP -> TM-INNER-FAILED (counted per run, recorded, listed; the step is taken with the last delta).
    alpha = 0: explicit step (no inner iteration). Run stop: direction-converged rule (A2.1), cap TM_CAP."""
    N = prob.N
    eta = c * N / st.lam1
    th = st.th0.clone()
    dvec = (t - st.F0) / vnorm(t - st.F0)
    dir_abs, dir_win, dir_run = CFG["DIR_STOP_REL"] * s, int(CFG["DIR_WINDOW"]), 0
    k, status = 0, None
    n_cap, inner_tot, inner_max, worst_g_ratio = 0, 0, 0, 0.0
    n_floor, floor_g_max = 0, 0.0
    capped_steps = []
    I = torch.eye(N, dtype=DT, device=th.device)
    while True:
        r = prob.F(th) - t
        rn = vnorm(r)
        if not math.isfinite(rn) or rn > 1e8 * max(s, 1e-300):
            status = "DIVERGED"
            break
        dir_run = dir_run + 1 if abs(float(dvec @ r)) <= dir_abs else 0
        if dir_run >= dir_win:
            status = "CONVERGED"
            break
        if k >= CFG["TM_CAP"]:
            status = "UNCONVERGED"
            break
        if alpha == 0:
            _, cache = prob.fwd(th)
            dl = -(eta / N) * prob.vjp(cache, r)
        else:
            a = eta * alpha / N
            dl = torch.zeros_like(th)
            ok, it = False, 0
            thn = vnorm(th)                          # A3.2: |theta| at the START of the step
            floor = CFG["TM_INNER_FP"] * FP_EPS * (1.0 + c) * thn
            while True:
                the = th + alpha * dl
                Je = prob.J(the)
                g = dl + (eta / N) * (Je.T @ (prob.F(the) - t))
                gn, tol = vnorm(g), CFG["TM_INNER_TOL_REL"] * vnorm(dl) + CFG["TM_INNER_TOL_ABS"]
                if gn <= tol:                        # (i)
                    ok = True
                    break
                if gn <= floor:                      # (ii) A3.2 rounding floor
                    ok = True
                    n_floor += 1
                    floor_g_max = max(floor_g_max, gn / (FP_EPS * thn))
                    break
                if it >= CFG["TM_INNER_CAP"]:
                    break
                Kt = gram(Je) + I / a
                dl = dl - (g - Je.T @ csolve(chol(Kt), Je @ g))
                it += 1
                Je = None
            inner_tot += it
            inner_max = max(inner_max, it)
            if not ok:
                n_cap += 1
                worst_g_ratio = max(worst_g_ratio, gn / tol)
                if len(capped_steps) < 20:
                    capped_steps.append(k)
        th = th + dl
        k += 1
        if k % 100 == 0:
            heartbeat(f"TM({c:g},{alpha:g}) step {k} |d.(F-t)|={abs(float(dvec @ r)):.3e} (target {dir_abs:.3e}) inner caps {n_cap}")
    return dict(theta=th, status=status, steps=k, final_res=rn, final_res_over_s=rn / s,
                dir_res_over_s=abs(float(dvec @ r)) / s, tm_inner_capped=n_cap, tm_inner_total=inner_tot,
                tm_inner_max=inner_max, tm_worst_g_over_tol=worst_g_ratio, tm_capped_steps=capped_steps,
                tm_inner_status="TM-INNER-FAILED" if n_cap else "OK", tm_inner_floor_accepts=n_floor,
                tm_floor_g_max=floor_g_max)


def run_SL_logged(prob, st, t, m, log, keep_points=False):
    """NEW (Part R). SL(m) with step log: theta_e = theta_k, b = K(theta_k) a, a from jump_a (same arithmetic as run_SL:
    jump_a is a copy of jump and J(theta_k) is passed as Jb, which is what jump computes internally)."""
    th = st.th0.clone()
    status, itc, itn, nfail = "CONVERGED", 0, 0, 0
    for k in range(1, m + 1):
        tk = st.F0 + (k / m) * (t - st.F0)
        Jb = prob.J(th)
        jr = jump_a(prob, th, tk, Jb=Jb)
        item = dict(Fe=prob.F(th).clone(), b=gram(Jb) @ jr["a"])
        if keep_points:
            item["the"] = th.clone()
        log.append(item)
        Jb = None
        th = jr["theta"]
        itc += jr["it_chord"]
        itn += jr["it_newton"]
        if jr["status"] != "CONVERGED":
            nfail += 1
            status = "JUMP-FAILED"
    return dict(theta=th, status=status, steps=m, jump_it_chord=itc, jump_it_newton=itn, jumps_failed=nfail)


def run_DJ_logged(prob, st, t, c, log, keep_points=False):
    """NEW (SPEC9 6.2). Same arithmetic as run_DJ plus the log: theta_e = theta_k, b = -c (F(theta_k) - t)."""
    th = st.th0.clone()
    target = CFG["DJ_TOL_REL"] * vnorm(t - st.F0)
    status, k = None, 0
    while True:
        F = prob.F(th)
        r = F - t
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
        item = dict(Fe=F.clone(), b=-c * r)
        if keep_points:
            item["the"] = th.clone()
        log.append(item)
        th = th - c * (J.T @ csolve(chol(gram(J)), r))
        k += 1
    return dict(theta=th, status=status, steps=k, final_res=rn)


def L_AJ(alpha):
    return dict(kind="AJ", alpha=float(alpha))


def L_SLa(m, alpha):
    return dict(kind="SLa", m=int(m), alpha=float(alpha))


def L_TJ(c, alpha):
    return dict(kind="TJ", c=float(c), alpha=float(alpha))


def L_TM(c, alpha):
    return dict(kind="TM", c=float(c), alpha=float(alpha))


def learner_name9(L):
    """NEW: names of the SPEC9 learners; SPEC8 learners keep their SPEC8 names (learner_name)."""
    k = L["kind"]
    if k == "AJ":
        return f"AJ({L['alpha']:g})"
    if k == "SLa":
        return f"SLa({L['m']},{L['alpha']:g})"
    if k == "TJ":
        return f"TJ({L['c']:g},{L['alpha']:g})"
    if k == "TM":
        return f"TM({L['c']:g},{L['alpha']:g})"
    return learner_name(L)


def predict_f9(L, lam_ratio=None):
    """NEW: closed-form f of SPEC9 (AJ: alpha; SLa: 1/2 + (alpha - 1/2)/m; TJ: (1+k)/(2+k), k = c(2 alpha - 1);
    TM on eigen-direction i: (1+k_i)/(2+k_i), k_i = c_i (2 alpha - 1), c_i = c lambda_i/lambda_1); SPEC8 learners: predict_f."""
    k = L["kind"]
    if k == "AJ":
        return L["alpha"]
    if k == "SLa":
        return 0.5 + (L["alpha"] - 0.5) / L["m"]
    if k == "TJ":
        kap = L["c"] * (2 * L["alpha"] - 1)
        return (1 + kap) / (2 + kap)
    if k == "TM":
        if lam_ratio is None:
            return None
        kap = L["c"] * lam_ratio * (2 * L["alpha"] - 1)
        return (1 + kap) / (2 + kap)
    return predict_f(L, lam_ratio)


def run_learner9(prob, st, t, s, L):
    k = L["kind"]
    if k == "AJ":
        return run_AJ(prob, st, t, L["alpha"])
    if k == "SLa":
        return run_SLa(prob, st, t, L["m"], L["alpha"])
    if k == "TJ":
        return run_TJ(prob, st, t, L["c"], L["alpha"])
    if k == "TM":
        return run_TM(prob, st, t, s, L["c"], L["alpha"])
    return run_learner(prob, st, t, s, L)


def run_logged(prob, st, t, L, log, keep_points=False):
    """Part R learners with the step log (F_e, b)."""
    k = L["kind"]
    if k == "SL":
        return run_SL_logged(prob, st, t, L["m"], log, keep_points)
    if k == "DJ":
        return run_DJ_logged(prob, st, t, L["c"], log, keep_points)
    if k == "AJ":
        return run_AJ(prob, st, t, L["alpha"], log=log, keep_points=keep_points)
    if k == "SLa":
        return run_SLa(prob, st, t, L["m"], L["alpha"], log=log, keep_points=keep_points)
    raise ValueError(k)


def usable9(sc, need_converged=False, supp=False):
    """NEW: why a single Part A / G run is excluded ('' = usable): the SPEC9 failure statuses first, then the reviewed usable().
    supp=True: SUPPLEMENTARY reading, the run status with hz evaluated on the step vector (learner_status_supp)."""
    if supp and "learner_status_supp" in sc:
        sc = dict(sc, learner_status=sc["learner_status_supp"])
    st = sc.get("learner_status")
    if st in AJ_BAD or st in ("ERROR", "SKIPPED"):
        return st
    return usable(sc, need_converged)



# =============================================================================================================
# SPEC9: training scalars, held-out data (Part G), combinations, starts, context
# =============================================================================================================
_SCAL = {}


def train_scalars(seed, n):
    """NEW. The two scalars (mean, std) with which get_data(seed, n) standardised the training inputs, recomputed by the
    SAME operations on the SAME selection (so they are bit-identical; unit test V10 checks this against get_data)."""
    key = (int(seed), int(n))
    if key in _SCAL:
        return _SCAL[key]
    raw = _load_mnist_raw()
    g = gen(1000 + seed)
    if isinstance(raw, str):
        x = torch.randn(n, 196, generator=g, dtype=DT)
    else:
        data, targets = raw
        perm = torch.randperm(data.shape[0], generator=g)
        idx = perm[:n]
        img = data[idx].to(DT).div(255.0).unsqueeze(1)
        x = Fnn.avg_pool2d(img, 2).reshape(n, 196)
    m, sd = x.mean(), x.std(unbiased=False)
    _SCAL[key] = (x, m, sd)
    return _SCAL[key]


_MNIST_TEST_RAW = None


def _load_mnist_test_raw():
    """NEW. MNIST TEST split (10 000 images, dataset order), or 'synthetic' when the training data are synthetic."""
    global _MNIST_TEST_RAW
    if _MNIST_TEST_RAW is not None:
        return _MNIST_TEST_RAW
    raw = _load_mnist_raw()
    if isinstance(raw, str):
        _MNIST_TEST_RAW = "synthetic"
        return _MNIST_TEST_RAW
    force = CFG.get("FORCE_DATA_SOURCE")
    ds = torchvision.datasets.MNIST(root=CFG.get("MNIST_ROOT", "./mnist_data"), train=False,
                                    download=bool(CFG.get("MNIST_DOWNLOAD", True)) and force is None)
    _MNIST_TEST_RAW = (ds.data.clone(), ds.targets.clone())
    return _MNIST_TEST_RAW


_TEST = {}


def get_test_data(seed, n):
    """NEW (Part G). Held-out inputs preprocessed like the training inputs (2x2 average pooling, flatten) and standardised
    with the mean and std of THIS seed's TRAINING selection (train_scalars(seed, n)). MNIST: the full test split in dataset
    order; synthetic fallback: TEST_SYNTH_N Gaussian inputs from gen(5000 + seed) with random labels.
    Returns dict(X, labels, Y (flat one-hot), mean, std)."""
    key = (int(seed), int(n))
    if key in _TEST:
        return _TEST[key]
    _, m, sd = train_scalars(seed, n)
    raw = _load_mnist_test_raw()
    C = CFG["N_CLASSES"]
    if isinstance(raw, str):
        g = gen(5000 + seed)
        nt = int(CFG["TEST_SYNTH_N"])
        x = torch.randn(nt, 196, generator=g, dtype=DT)
        labels = torch.randint(0, C, (nt,), generator=g)
    else:
        data, targets = raw
        img = data.to(DT).div(255.0).unsqueeze(1)
        x = Fnn.avg_pool2d(img, 2).reshape(data.shape[0], 196)
        labels = targets.long()
    x = (x - m) / sd
    Y = Fnn.one_hot(labels, C).to(DT).reshape(-1)
    out = dict(X=x.to(DEVICE), labels=labels.to(DEVICE), Y=Y.to(DEVICE), mean=m, std=sd, n_test=int(x.shape[0]))
    _TEST[key] = out
    return out


def test_outputs(prob, X, th):
    """F_test(theta): network outputs on held-out inputs, float64, forward pass only, in chunks of TEST_CHUNK inputs."""
    ch = int(CFG["TEST_CHUNK"])
    with torch.no_grad():
        outs = [mlp_forward(th, X[i:i + ch], prob.shapes, prob.act)[0] for i in range(0, X.shape[0], ch)]
    return torch.cat(outs)


def test_loss_acc(Ft, td):
    """Test loss l = |F_test - Y_test|^2 / (2 N_test) and accuracy (argmax == label)."""
    C = CFG["N_CLASSES"]
    r = Ft - td["Y"]
    loss = float(r @ r) / (2.0 * r.numel())
    acc = float((Ft.view(-1, C).argmax(1) == td["labels"]).to(DT).mean())
    return loss, acc


def cname9(cb):
    """Combination name: SPEC8 combo_name, plus '-n<n>' when the training-set size differs from N_TRAIN (G4, n = 128)."""
    nm = combo_name(cb)
    n = int(cb.get("n", CFG["N_TRAIN"]))
    return nm if n == CFG["N_TRAIN"] else f"{nm}-n{n}"


def combo_problem9(cb):
    """combo_problem with the training-set size n of the combination (default N_TRAIN, identical to combo_problem);
    for n > N_TRAIN the Jacobian is computed in chunks of G4_JAC_CHUNK_BIG rows (SPEC9 10.6)."""
    n = int(cb.get("n", CFG["N_TRAIN"]))
    if n == CFG["N_TRAIN"]:
        return combo_problem(cb)
    X, labels, Y = get_data(cb["seed"], n)
    sizes = [196, CFG["WIDTH"], CFG["WIDTH"], CFG["N_CLASSES"]]
    keep = CFG.get("JAC_CHUNK")
    CFG["JAC_CHUNK"] = CFG["G4_JAC_CHUNK_BIG"]
    try:
        prob, th = layered_problem(X, sizes, cb["act"], cb["seed"], name=cname9(cb))
    finally:
        CFG["JAC_CHUNK"] = keep
    return prob, th, Y, labels


SCIENCE_KEYS9 = ["MODE", "N_TRAIN", "WIDTH", "N_CLASSES", "JUMP_TOL", "JUMP_CHORD_CAP", "JUMP_NEWTON_CAP", "DJ_TOL_REL",
                 "DJ_CAP", "NP_RES_TOL", "NP_STAT_TOL", "NP_CAP", "NP_MIN_STEP", "NP_MERIT", "NP_DIST_SLACK", "FIT_C",
                 "FIT_MSE", "FIT_CAP", "ILL_COND", "FLOOR_FACTOR", "FLOOR_REL", "S1_FRAC", "S2_FRAC", "DIR_STOP_REL",
                 "DIR_WINDOW", "AJ_OUT_TOL_REL", "AJ_OUT_TOL_ABS", "AJ_OUTER_CAP", "AJ_HZ_TOL", "TJ_TOL_REL", "TJ_CAP",
                 "TM_INNER_TOL_REL", "TM_INNER_TOL_ABS", "TM_INNER_CAP", "TM_CAP", "A_COMBOS", "A_DIRS", "TM_DIRS",
                 "AJ_ALPHAS", "SLA_SET", "TJ_SET", "TM_SET", "ANCHORS", "TEST_SYNTH_N", "G4_COMBOS", "G4_TAUS",
                 "G4_LEARNERS", "G4_GD_STOP_REL", "G4_GD_CAP", "G4_NP_CAP", "G4_NP_TAUS", "R_SMOOTH", "R_RELU", "R_DIRS",
                 "R_SIZES_RELU", "R_SIZES_SMOOTH", "G_REF", "R_LEARNERS", "R_PRIMARY", "R_E_SIZE_MAX", "LIN_LEARNERS",
                 "AJ_HZ_FP", "TM_INNER_FP"]


def config_signature():
    """Signature of every constant that changes the numbers (NOT the ANALYSIS_ONLY flag, NOT engineering settings)."""
    d = {k: CFG.get(k) for k in SCIENCE_KEYS9}
    d["data_source"] = DATA_SOURCE
    d["dtype"] = "float64"
    d["spec"] = "SPEC9"
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


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


def start_paths(run_dir, cname):
    d = run_subdir(run_dir, "starts")
    return os.path.join(d, cname + ".pt"), os.path.join(d, cname + ".json")


def build_or_load_start(cb, run_dir, verbose=True, load_only=False):
    """Builds (or loads) theta0 of a combination: freshly initialised or DJ(0.5)-fitted to the true labels (SPEC8 sec. 2).
    Stores spectrum, directions, S0, S_true and the training scalars (mean, std as floats and float.hex)."""
    cname = cname9(cb)
    pt, js = start_paths(run_dir, cname)
    sig = config_signature()
    if os.path.exists(pt) and os.path.exists(js):
        info = read_json(js)
        if info.get("signature") == sig:
            if verbose:
                say(f"  {cname}: loaded cached start (built {info.get('built')})")
            return info
    if load_only:
        if verbose:
            say(f"  {cname}: no cached start (ANALYSIS_ONLY: not built)")
        return None
    t1 = time.time()
    n = int(cb.get("n", CFG["N_TRAIN"]))
    prob, th, Y, labels = combo_problem9(cb)
    info = dict(combo=cname, act=cb["act"], start=cb["start"], seed=cb["seed"], n=n, signature=sig,
                data_source=DATA_SOURCE, built=time.strftime("%Y-%m-%d %H:%M:%S"))
    if cb["start"] == "trained":
        th, fi = fit_dj(prob, th, Y)
        info.update(fi)
    st = Start(prob, th)
    info.update(st.spectrum())
    if not st.ill:
        dirs, dinfo, S0 = make_directions(st, cb["seed"], n)
        info.update(directions=dinfo, S0=S0, S_true=vnorm(Y - st.F0))
    _, m, sd = train_scalars(cb["seed"], n)
    info.update(train_mean=float(m), train_std=float(sd), train_mean_hex=float(m).hex(), train_std_hex=float(sd).hex())
    info.update(N=prob.N, P=prob.P, build_s=time.time() - t1)
    torch.save(th.detach().cpu(), pt)
    write_json_atomic(js, info)
    if verbose:
        say(f"  {cname}: built in {fmt_time(time.time() - t1)}")
    del st
    free_memory()
    return info


class Ctx9:
    """Everything a combination's cases share: problem, Start (J0, K0 factor, eigen-decomposition), directions
    (eigA-D, label, rand; 'true' = (Y_true - F0)/S_true), held-out data and F_test(theta0) (lazily)."""

    def __init__(self, cb, run_dir=None, prob=None, th0=None, Y=None, test=None):
        self.cb, self.cname = cb, cname9(cb) if "act" in cb else cb.get("name", "test")
        self.n = int(cb.get("n", CFG["N_TRAIN"]))
        if prob is None:
            self.prob, _, self.Y, self.labels = combo_problem9(cb)
            pt, _ = start_paths(run_dir, self.cname)
            th0 = torch.load(pt, map_location="cpu").to(DT).to(DEVICE)
        else:
            self.prob, self.Y, self.labels = prob, Y, None
        self.st = Start(self.prob, th0, tag=self.cname)
        self.dirs, self.dinfo, self.S0 = make_directions(self.st, cb.get("seed", 0), self.n)
        self.S = {k: self.S0 for k in self.dirs}
        if self.Y is not None:
            self.S_true = vnorm(self.Y - self.st.F0)
            self.dirs["true"] = (self.Y - self.st.F0) / self.S_true
            self.dinfo["true"] = dict(index=None, lam_ratio=None)
            self.S["true"] = self.S_true
        self.run_dir = run_dir
        self._test = test
        self._t0 = None

    def test(self):
        if self._test is None:
            self._test = get_test_data(self.cb["seed"], self.n)
        return self._test

    def test0(self):
        """F_test(theta0), l(theta0), acc(theta0) (once per combination)."""
        if self._t0 is None:
            td = self.test()
            F = test_outputs(self.prob, td["X"], self.st.th0)
            l, a = test_loss_acc(F, td)
            self._t0 = (F, l, a)
        return self._t0


# =============================================================================================================
# SPEC9 sec. 6: Part R - reference path, field W(tau), logged learners, general (reference-path) form of the law
# =============================================================================================================
class PredFailed(RuntimeError):
    pass


def preact_pattern(prob, th):
    """Signs (z > 0) of every hidden (unit, input) pre-activation at theta (all activations; for tanh/gelu stored only
    for completeness)."""
    with torch.no_grad():
        _, cache = mlp_forward(th, prob.X, prob.shapes, prob.act, keep=True)
    return torch.cat([(cache[i][2] > 0).reshape(-1) for i in range(len(cache) - 1)])


class RefPath:
    """Reference path on the line F0 + tau s d (d = SIGNED unit direction of the case, s = its size): grid
    tau_j = 2 j / G (j = 0..G, tau in [0, 2]); theta_ref[0] = theta0, theta_ref[j] = JUMP(theta_ref[j-1], F0 + tau_j s d).
    Only weight vectors are stored. Exact position theta_ref(tau) := JUMP(theta_ref[j], F0 + tau s d), j = floor(tau G / 2)
    (j = G -> G - 1); tau outside [0, 2] is clamped and flagged OUT-OF-RANGE.
    Field W(tau) b := V0 J(theta_ref(tau))^T K(theta_ref(tau))^{-1} b (K factorised at that point)."""

    def __init__(self, prob, st, d, s, G):
        self.prob, self.st, self.d, self.s, self.G = prob, st, d, s, int(G)
        self.th = [st.th0.clone()]
        self.status, self.fail_j = "OK", None
        pat0 = preact_pattern(prob, st.th0)
        prev = pat0
        self.nsw, self.cell_change = [0], []
        for j in range(1, self.G + 1):
            tau = 2.0 * j / self.G
            jr = jump(prob, self.th[-1], st.F0 + tau * s * d)
            if jr["status"] != "CONVERGED":
                self.status, self.fail_j = "REF-FAILED", j
                break
            self.th.append(jr["theta"])
            pat = preact_pattern(prob, jr["theta"])
            self.nsw.append(int((pat != pat0).sum()))
            self.cell_change.append(bool((pat != prev).any()))
            prev = pat
            if j % 16 == 0:
                heartbeat(f"reference path {j}/{self.G}")
        self._cache = {}
        self.n_pos_jumps = 0

    def complete(self):
        return self.status == "OK"

    def pos(self, tau):
        """(theta_ref(tau), out_of_range flag)."""
        oor = not (0.0 <= tau <= 2.0)
        tc = min(max(float(tau), 0.0), 2.0)
        if tc in self._cache:
            return self._cache[tc], oor
        j = min(int(math.floor(tc * self.G / 2.0)), self.G - 1)
        jr = jump(self.prob, self.th[j], self.st.F0 + tc * self.s * self.d)
        self.n_pos_jumps += 1
        if jr["status"] != "CONVERGED":
            raise PredFailed(f"JUMP to the exact reference position tau={tc!r} from grid point {j} failed")
        if len(self._cache) >= 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[tc] = jr["theta"]
        return jr["theta"], oor

    def W(self, tau, B):
        """V0 J(theta_ref(tau))^T K(theta_ref(tau))^{-1} B for a vector or the columns of a matrix B."""
        th, oor = self.pos(tau)
        st = self.st
        if torch.equal(th, st.th0):
            X = st.J0.T @ csolve(st.L0, B)
        else:
            J = self.prob.J(th)
            X = J.T @ csolve(chol(gram(J)), B)
            J = None
        return st.V0(X), oor

    def switch_on(self, T):
        """Any pre-activation sign change in a cell of the grid that intersects [0, T]."""
        return any(self.cell_change[j] for j in range(len(self.cell_change)) if 2.0 * j / self.G < T)


def R_profile(ref):
    """v_PP = W(1)(s d); omega(tau) = <W(tau)(s d), v_PP>/|v_PP|^2 and omega_perp(tau) = |W(tau)(s d) - omega v_PP|/|v_PP|
    at tau = k/16, k = 0..32 (grid points theta_ref[k G/32])."""
    sd = ref.s * ref.d
    vPP, _ = ref.W(1.0, sd)
    n2 = float(vPP @ vPP)
    om, omp = [], []
    for k in range(33):
        w, _ = ref.W(k / 16.0, sd)
        o = float(w @ vPP) / n2 if n2 > 0 else float("nan")
        om.append(o)
        omp.append(vnorm(w - o * vPP) / math.sqrt(n2) if n2 > 0 else float("nan"))
    return vPP, om, omp


def R_learner(prob, st, ref, vPP, t, L, keep_points=False):
    """Runs one logged learner on the case and evaluates the three predictions (P-field vector, P-omega scalar, P-smooth).
    Returns (scalars, vectors)."""
    s, d = ref.s, ref.d
    c0 = dict(prob.counts)
    t1 = time.time()
    log = []
    res = run_logged(prob, st, t, L, log, keep_points=keep_points)
    thL = res["theta"]
    sc = dict(learner=learner_name9(L), learner_status=res["status"], steps=res.get("steps"), n_log=len(log),
              f_smooth=predict_f9(L))
    for k, v in res.items():
        if k not in ("theta", "status", "steps") and not torch.is_tensor(v):
            sc["learner_" + k] = v
    if res["status"] == "DIVERGED" or not bool(torch.isfinite(thL).all()):
        sc["learner_status"] = "DIVERGED"
        return sc, {}
    vL = st.V0(thL - st.th0)
    n2 = float(vPP @ vPP)
    sc["rho_PP"] = float(vL @ vPP) / n2
    vpred = torch.zeros_like(vL)
    rho_om, oor, taus, betas, offl, omk = 0.0, 0, [], [], 0.0, []
    bperp = 0.0                                          # A3.4: max_k |b_k - beta_k s d| / |b_k|
    sd = s * d
    for item in log:
        Fe, b = item["Fe"], item["b"]
        tau = float(d @ (Fe - st.F0)) / s                 # SIGNED direction, outputs AT THE EVALUATION POINT
        beta = float(d @ b) / s
        X, flag = ref.W(tau, torch.stack([b, sd], 1))
        oor += int(flag)
        vpred = vpred + X[:, 0]
        o = float(X[:, 1] @ vPP) / n2
        rho_om += beta * o
        taus.append(tau)
        betas.append(beta)
        omk.append(o)
        offl = max(offl, vnorm((Fe - st.F0) - tau * sd) / s)
        nb = vnorm(b)
        bperp = max(bperp, vnorm(b - beta * sd) / nb if nb > 0 else 0.0)
        heartbeat(f"{sc['learner']} prediction step {len(taus)}/{len(log)}")
    mt = max(taus) if taus else 0.0
    nL, nP = vnorm(vL), vnorm(vpred)
    sc.update(cosF=cosine(vL, vpred), nrF=nL / nP if nP > 0 else float("nan"),
              rho_field=float(vpred @ vPP) / n2, rho_omega=rho_om, out_of_range=oor, bperp_max=bperp,
              rho_perp=float(vpred @ vPP) / n2 - rho_om, max_tau=mt,
              min_tau=min(taus) if taus else 0.0, offline_max=offl, sum_beta=sum(betas),
              nsw_end=int((preact_pattern(prob, thL) != preact_pattern(prob, st.th0)).sum()),
              ref_switch_01=ref.switch_on(1.0), ref_switch_0maxtau=ref.switch_on(mt),
              ref_switch_0max1tau=ref.switch_on(max(1.0, mt)), nvL=nL, nvpred=nP,
              n_J=prob.counts["J"] - c0["J"], n_F=prob.counts["F"] - c0["F"], t_learner_pred=time.time() - t1)
    if len(taus) <= 64:
        sc.update(taus=taus, betas=betas, omega_k=omk)
    vec = dict(vL=vL, vpred=vpred)
    if keep_points:
        vec["log"] = log
        vec["thL"] = thL
    return sc, vec



# =============================================================================================================
# SPEC9: cases (one case = one JSON file in done/ + one vector file in vecs/), plan, running, workers
# =============================================================================================================
def L_from_spec(x):
    """Learner dict from a CONFIG entry ("AJ", alpha) / ("SLa", m, alpha) / ("TJ", c, alpha) / ("TM", c, alpha) /
    ("SL", m) / ("DJ", c) / ("GD", c)."""
    k = x[0]
    if k == "AJ":
        return L_AJ(x[1])
    if k == "SLa":
        return L_SLa(x[1], x[2])
    if k == "TJ":
        return L_TJ(x[1], x[2])
    if k == "TM":
        return L_TM(x[1], x[2])
    if k == "SL":
        return L_SL(int(x[1]))
    if k == "DJ":
        return L_DJ(float(x[1]))
    if k == "GD":
        return L_GD(float(x[1]), CFG["G4_GD_STOP_REL"], CFG["G4_GD_CAP"])
    raise ValueError(x)


def A_learners(with_tm):
    Ls = [L_AJ(a) for a in CFG["AJ_ALPHAS"]] + [L_SLa(m, a) for m, a in CFG["SLA_SET"]]
    Ls += [L_TJ(c, a) for c, a in CFG["TJ_SET"]] + [L_from_spec(x) for x in CFG["ANCHORS"]]
    if with_tm:
        Ls += [L_TM(c, a) for c, a in CFG["TM_SET"]]
    return Ls


def _case9(cname, part, grp, L, d, size_lbl, frac, sign, **kw):
    ln = learner_name9(L) if L is not None else "ref"
    nm = f"{cname}__{grp}__{ln}__{d}__{size_lbl}__{'p' if sign > 0 else 'm'}"
    c = dict(name=nm, combo=cname, part=part, grp=grp, kind=grp, L=L, learner=ln, dir=d, size=size_lbl, frac=frac,
             sign=sign)
    c.update(kw)
    return c


def plan_A(cb):
    """Part A (with G1-G3 measured in the same cases): six directions (+ 'true' for start = init), sizes s1, s2, both signs."""
    cname = cname9(cb)
    dirs = list(CFG["A_DIRS"]) + (["true"] if cb["start"] == "init" else [])
    cases = []
    for d in dirs:
        for sl, fr in (("s1", CFG["S1_FRAC"]), ("s2", CFG["S2_FRAC"])):
            for sg in (1, -1):
                for L in A_learners(with_tm=d in CFG["TM_DIRS"]):
                    cases.append(_case9(cname, "A", "A", L, d, sl, fr, sg))
    return cases


def plan_lin():
    cb = linear_combo()
    cname = cname9(cb)
    return [_case9(cname, "lin", "lin", L_from_spec(x), d, "s1", CFG["S1_FRAC"], sg)
            for d in ("label", "rand") for sg in (1, -1) for x in CFG["LIN_LEARNERS"]]


def plan_G4(cb):
    cname = cname9(cb)
    cases = []
    for tau in CFG["G4_TAUS"]:
        for x, fnom in CFG["G4_LEARNERS"]:
            c = _case9(cname, "G4", "G4", L_from_spec(x), "true", f"tau{tau:g}", tau, 1, f_nominal=fnom)
            cases.append(c)
    return cases


def plan_R(cb):
    """Part R: directions R_DIRS, sizes (smaller first), both signs, every sign a separate case; all learners of the case
    share one reference path (one case = one JSON with every learner)."""
    cname = cname9(cb)
    sizes = CFG["R_SIZES_RELU"] if cb["act"] == "relu" else CFG["R_SIZES_SMOOTH"]
    return [_case9(cname, "R", "R", None, d, f"r{fr:g}", fr, sg) for fr in sorted(sizes) for d in CFG["R_DIRS"]
            for sg in (1, -1)]


def combos(kind):
    if kind == "A":
        return [dict(act=a, start=s, seed=sd) for a, s, sd in CFG["A_COMBOS"]]
    if kind == "G4":
        return [dict(act=a, start="init", seed=sd, n=n) for a, sd, n in CFG["G4_COMBOS"]]
    if kind == "R_smooth":
        return [dict(act=a, start=s, seed=sd) for a, s, sd in CFG["R_SMOOTH"]]
    if kind == "R_relu":
        return [dict(act=a, start=s, seed=sd) for a, s, sd in CFG["R_RELU"]]
    raise ValueError(kind)


PARTS = ["lin", "A", "G4", "R_smooth", "R_relu"]      # SPEC9 10.2 order of work


def make_plan(ill=()):
    """{part: [dict(combo, cases[, kernel_only])]} in the order of work; ILL-CONDITIONED combinations are skipped."""
    plan = {"lin": [dict(combo=linear_combo(), cases=plan_lin(), kernel_only=True)]}
    plan["A"] = [dict(combo=cb, cases=plan_A(cb)) for cb in combos("A") if cname9(cb) not in ill]
    plan["G4"] = [dict(combo=cb, cases=plan_G4(cb)) for cb in combos("G4") if cname9(cb) not in ill]
    plan["R_smooth"] = [dict(combo=cb, cases=plan_R(cb)) for cb in combos("R_smooth") if cname9(cb) not in ill]
    plan["R_relu"] = [dict(combo=cb, cases=plan_R(cb)) for cb in combos("R_relu") if cname9(cb) not in ill]
    return plan


def all_start_combos():
    out, seen = [], set()
    for cb in [linear_combo()] + combos("A") + combos("G4") + combos("R_smooth") + combos("R_relu"):
        nm = cname9(cb)
        if nm not in seen:
            seen.add(nm)
            out.append(cb)
    return out


# ------------------------------------------------------------------ case runners
def _test_fields(ctx, prefix, th):
    td = ctx.test()
    F = test_outputs(ctx.prob, td["X"], th)
    l, a = test_loss_acc(F, td)
    return F, {f"loss_{prefix}": l, f"acc_{prefix}": a}


def run_A_case(ctx, case):
    """Part A / G1-G3 / linear control: learner, then measure_ext (SPEC8 measurement of the SAME fibre) and the held-out
    quantities G_L = F_test(theta_L) - F_test(theta_TH), G_NP = F_test(theta_NP) - F_test(theta_TH)."""
    prob, st = ctx.prob, ctx.st
    L = case["L"]
    s = case["frac"] * ctx.S[case["dir"]]
    t = st.F0 + case["sign"] * s * ctx.dirs[case["dir"]]
    rec = dict(s=s, S_dir=ctx.S[case["dir"]])
    c0 = dict(prob.counts)
    t1 = time.time()
    res = run_learner9(prob, st, t, s, L)
    t2 = time.time()
    thL = res["theta"]
    for k, v in res.items():
        if k in ("theta", "status") or torch.is_tensor(v):
            continue
        rec["learner_" + k if k not in ("steps",) else k] = v
    rec["learner_status"] = res["status"]
    vecs = {}
    if res["status"] == "DIVERGED" or res["status"] == "AJ-FAILED" or not bool(torch.isfinite(thL).all()):
        rec.update(np_status="SKIPPED")
        if res["status"] != "AJ-FAILED":
            rec["learner_status"] = "DIVERGED"
    else:
        sc, vL, vNP, thNP, thTH = measure_ext(prob, st, thL)
        rec.update(sc)
        vecs.update(vL=vL, vNP=vNP)
        F0t, l0, a0 = ctx.test0()
        FL, d1 = _test_fields(ctx, "L", thL)
        FT, d2 = _test_fields(ctx, "TH", thTH)
        FN, d3 = _test_fields(ctx, "NP", thNP)
        rec.update(d1)
        rec.update(d2)
        rec.update(d3)
        rec.update(loss_0=l0, acc_0=a0)
        GL, GNP = FL - FT, FN - FT
        vecs.update(GL=GL, GNP=GNP)
        dth = vnorm(FT - F0t)
        rec.update(nGL=vnorm(GL), nGNP=vnorm(GNP), GNP_over_dTH=vnorm(GNP) / dth if dth > 0 else float("nan"),
                   GL_over_dL=vnorm(GL) / max(vnorm(FL - F0t), 1e-300), dloss_L=d1["loss_L"] - d2["loss_TH"],
                   n_test=ctx.test()["n_test"])
        if L["kind"] == "AJ" and L["alpha"] == 1.0 and case["grp"] == "A":
            np0 = nearest_point(prob, st.th0, prob.F(thL), st.th0)     # HA6: second solver, started from theta0
            dn0 = vnorm(np0["theta"] - st.th0)
            rec.update(ha6_np0_status=np0["status"], ha6_np0_it=np0["it"],
                       ha6_rel=vnorm(thL - np0["theta"]) / dn0 if dn0 > 0 else float("nan"))
    t3 = time.time()
    rec.update(t_learner=t2 - t1, t_measure=t3 - t2, n_J=prob.counts["J"] - c0["J"], n_F=prob.counts["F"] - c0["F"],
               n_VJP=prob.counts["VJP"] - c0["VJP"])
    return rec, vecs


def run_G4_case(ctx, case):
    """G4: large move toward the true labels, t = F0 + tau (Y_true - F0), sign + only."""
    prob, st = ctx.prob, ctx.st
    L, tau = case["L"], case["frac"]
    t = st.F0 + tau * (ctx.Y - st.F0)
    s = vnorm(t - st.F0)
    rec = dict(s=s, tau=tau)
    c0 = dict(prob.counts)
    t1 = time.time()
    res = run_learner9(prob, st, t, s, L)
    thL = res["theta"]
    for k, v in res.items():
        if k in ("theta", "status") or torch.is_tensor(v):
            continue
        rec["learner_" + k if k not in ("steps",) else k] = v
    rec["learner_status"] = res["status"]
    if not bool(torch.isfinite(thL).all()):
        rec["learner_status"] = "DIVERGED"
        return rec, {}
    FLtr = prob.F(thL)
    rec["train_res"] = vnorm(FLtr - t) / s
    _, d1 = _test_fields(ctx, "L", thL)
    rec.update(d1)
    F0t, l0, a0 = ctx.test0()
    rec.update(loss_0=l0, acc_0=a0, dist=vnorm(thL - st.th0))
    lam1, trk = _lam1_tr(gram(prob.J(thL)))
    rec.update(lam1_L=lam1, trK_L=trk)
    vecs = {}
    if ctx.n == CFG["N_TRAIN"] and any(abs(tau - x) < 1e-12 for x in CFG["G4_NP_TAUS"]):
        sc, vL, vNP = measure(prob, st, thL, np_cap=CFG["G4_NP_CAP"])
        rec.update({"np_" + k if not k.startswith("np_") else k: v for k, v in sc.items()})
        vecs.update(vL=vL, vNP=vNP)
    rec.update(t_total=time.time() - t1, n_J=prob.counts["J"] - c0["J"], n_F=prob.counts["F"] - c0["F"])
    return rec, vecs


def run_R_case(ctx, case):
    """Part R case = (combination, direction, size, sign): reference path, profile, every learner, and (smooth controls,
    sizes <= R_E_SIZE_MAX) one nearest-point solve for report R-e. NO nearest-point solve otherwise."""
    prob, st = ctx.prob, ctx.st
    s = case["frac"] * ctx.S0
    d = case["sign"] * ctx.dirs[case["dir"]]                 # SIGNED unit direction of the case
    t = st.F0 + s * d
    c0 = dict(prob.counts)
    t1 = time.time()
    G = int(CFG["G_REF"])
    ref = RefPath(prob, st, d, s, G)
    rec = dict(s=s, G_REF=G, ref_status=ref.status, ref_fail_j=ref.fail_j, t_ref=time.time() - t1,
               n_J_ref=prob.counts["J"] - c0["J"])
    vecs = {}
    if not ref.complete():
        rec["learners"] = {}
        return rec, vecs
    rec.update(nsw_16=[ref.nsw[k * G // 32] for k in range(33)], cells_switch_01=int(sum(ref.cell_change[:G // 2])),
               cells_switch_02=int(sum(ref.cell_change)), cell_change=[int(x) for x in ref.cell_change],
               nsw_1=ref.nsw[G // 2])
    th1, _ = ref.pos(1.0)
    rec["dref1"] = vnorm(th1 - st.th0)
    vPP, om, omp = R_profile(ref)
    rec.update(omega=om, omega_perp=omp, nvPP=vnorm(vPP))
    vecs["vPP"] = vPP
    learners = {}
    for x in CFG["R_LEARNERS"]:
        L = L_from_spec(x)
        ln = learner_name9(L)
        _HB["label"] = f"{case['name']} {ln}:"
        try:
            sc, v = R_learner(prob, st, ref, vPP, t, L)
        except PredFailed as e:
            sc, v = dict(learner=ln, learner_status="PRED-FAILED", error=str(e)), {}
        learners[ln] = sc
        for k, vv in v.items():
            vecs[f"{k}__{ln}"] = vv
        free_memory()
    rec["learners"] = learners
    if prob.act != "relu" and case["frac"] <= CFG["R_E_SIZE_MAX"] * (1 + 1e-12):
        npr = nearest_point(prob, st.th0, t, st.th0)            # R-e only: SPEC8 nearest point of the nominal target
        vNP = st.V0(npr["theta"] - st.th0)
        rec.update(re_np_status=npr["status"], re_np_it=npr["it"], re_cos=cosine(vPP, vNP),
                   re_ratio=vnorm(vPP) / vnorm(vNP) if vnorm(vNP) > 0 else float("nan"))
        vecs["vNP_re"] = vNP
    rec.update(t_total=time.time() - t1, n_J=prob.counts["J"] - c0["J"], n_F=prob.counts["F"] - c0["F"],
               ref_pos_jumps=ref.n_pos_jumps)
    del ref
    return rec, vecs


def run_case9(ctx, case):
    st = ctx.st
    rec = dict(name=case["name"], combo=ctx.cname, act=ctx.cb["act"], start=ctx.cb["start"], seed=ctx.cb["seed"],
               n=ctx.n, part=case["part"], grp=case["grp"], kind=case["kind"], learner=case["learner"], L=case["L"],
               dir=case["dir"], size=case["size"], frac=case["frac"], sign=case["sign"], S0=ctx.S0,
               data_source=DATA_SOURCE, lam1_K0=st.lam1)
    if case["dir"] in ctx.dinfo:
        di = ctx.dinfo[case["dir"]]
        rec.update(dir_index=di["index"], lam_ratio=di["lam_ratio"])
    if case["L"] is not None:
        rec["f_pred"] = predict_f9(case["L"], rec.get("lam_ratio"))
    if "f_nominal" in case:
        rec["f_nominal"] = case["f_nominal"]
    if case["kind"] in ("A", "lin"):
        r, v = run_A_case(ctx, case)
    elif case["kind"] == "G4":
        r, v = run_G4_case(ctx, case)
    elif case["kind"] == "R":
        r, v = run_R_case(ctx, case)
    else:
        raise ValueError(case["kind"])
    rec.update(r)
    return rec, v


def run_combo_cases9(cb, cases, run_dir, sig, worker="kernel"):
    """Run every not-yet-done case of one combination (resumable). Returns (n_run, n_failed)."""
    todo = [c for c in cases if not is_done(run_dir, c["name"], sig)]
    if not todo:
        return 0, 0
    ddir, fdir, vdir = (run_subdir(run_dir, s) for s in ("done", "failed", "vecs"))
    cname = cname9(cb)
    progress(f"[{worker}] {cname}: {len(todo)} of {len(cases)} cases to run ({time.strftime('%H:%M:%S')})")
    ctx = Ctx9(cb, run_dir)
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
            rec, vecs = run_case9(ctx, case)
            rec.update(signature=sig, worker=worker, device=str(DEVICE), wall_s=time.time() - t1,
                       finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            save_vecs_atomic(os.path.join(vdir, nm + ".pt"), vecs)
            write_json_atomic(os.path.join(ddir, nm + ".json"), rec)
            nrun += 1
        except Exception:
            tb = traceback.format_exc()
            progress(f"\n!!!!! [{worker}] CASE FAILED: {case['name']}\n{tb}")
            write_json_atomic(fp, dict(name=case["name"], combo=cname, worker=worker, error=tb,
                                       time=time.strftime("%Y-%m-%d %H:%M:%S")))
            nfail += 1
        free_memory()
    progress(f"[{worker}] {cname}: finished {nrun} cases, {nfail} failed, in {fmt_time(time.time() - t_start)}")
    return nrun, nfail


# ------------------------------------------------------------------ workers (one process per GPU, static split)
def write_run_files(run_dir, items, part):
    cfg = dict(CFG)
    cfg.update(FORCE_DATA_SOURCE=DATA_SOURCE, MNIST_DOWNLOAD=False,
               MNIST_ROOT=os.path.abspath(CFG.get("MNIST_ROOT", "./mnist_data")), SIGNATURE=config_signature())
    write_json_atomic(os.path.join(run_dir, "config.json"), cfg)
    write_json_atomic(os.path.join(run_dir, f"plan_{part}.json"), items)


def worker_main(run_dir, k, n, device, threads, part):
    cfg = read_json(os.path.join(run_dir, "config.json"))
    configure(cfg)
    torch.set_num_threads(int(threads))
    set_device("cpu" if device == "cpu" else "cuda")
    items = read_json(os.path.join(run_dir, f"plan_{part}.json"))
    wid = f"w{k}-{'cpu' if device == 'cpu' else 'gpu' + str(device)}"
    say(f"[{wid}] part {part} pid {os.getpid()} device {DEVICE} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"threads {torch.get_num_threads()} ({time.strftime('%H:%M:%S')})")
    _load_mnist_raw()
    sig = config_signature()
    assert sig == cfg["SIGNATURE"], f"worker signature {sig} != kernel signature {cfg['SIGNATURE']}"
    for i, item in enumerate(items):
        if i % n != k:
            continue
        run_combo_cases9(item["combo"], item["cases"], run_dir, sig, worker=wid)
    say(f"[{wid}] finished ({time.strftime('%H:%M:%S')})")


def _spawn(run_dir, k, n, dev, threads, part):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "" if dev == "cpu" else str(dev)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = str(threads)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONIOENCODING"] = "utf-8"
    libdir = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-u", "-c",
           f"import sys; sys.path.insert(0, {libdir!r}); import {LIBNAME}; "
           f"{LIBNAME}.worker_main({run_dir!r}, {k}, {n}, {dev!r}, {int(threads)}, {part!r})"]
    logp = os.path.join(run_subdir(run_dir, "logs"), f"worker{k}_{part}.log")
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
    cpu_w = int(os.environ.get("PART9_CPU_WORKERS", CFG.get("CPU_WORKERS", 0)) or 0)
    if CFG.get("PARALLEL", True) and ng >= 2:
        env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ids = [x.strip() for x in env.split(",") if x.strip()] if env.strip() else [str(i) for i in range(ng)]
        return ids[:ng], f"{ng} GPUs: one worker per GPU"
    if cpu_w > 0:
        return ["cpu"] * cpu_w, f"{cpu_w} CPU worker processes (testing mode)"
    return [], f"{ng} GPU(s): in-kernel path"


def run_pool(items, run_dir, slots, part):
    write_run_files(run_dir, items, part)
    sig = config_signature()
    threads = torch.get_num_threads()
    n = len(slots)
    procs = [_spawn(run_dir, k, n, slots[k], threads, part) for k in range(n)]
    for k, P in enumerate(procs):
        mine = [cname9(it["combo"]) for i, it in enumerate(items) if i % n == k]
        progress(f"  part {part}: worker {k} on {'CPU' if P['dev'] == 'cpu' else 'GPU ' + str(P['dev'])}: pid {P['p'].pid}, "
                 f"combinations {mine}, log {P['logp']}")
    names = [c["name"] for it in items for c in it["cases"]]
    t0, last = time.time(), time.time()
    try:
        while True:
            time.sleep(2.0)
            alive = [P for P in procs if P["p"].poll() is None]
            if time.time() - last >= CFG.get("PROGRESS_EVERY_S", 60) or not alive:
                last = time.time()
                nd = sum(is_done(run_dir, nm, sig) for nm in names)
                nf = sum(os.path.exists(os.path.join(run_dir, "failed", safe_name(nm) + ".json")) for nm in names)
                progress(f"  [{time.strftime('%H:%M:%S')}] part {part} progress: {nd}/{len(names)} cases done, {nf} failed, "
                         f"{len(alive)}/{n} workers running, elapsed {fmt_time(time.time() - t0)}")
            if not alive:
                break
    except KeyboardInterrupt:
        progress("interrupted: terminating workers (finished cases stay on disk; re-run to resume)")
        for P in procs:
            if P["p"].poll() is None:
                P["p"].terminate()
        raise
    finally:
        for P in procs:
            P["log"].close()
    for k, P in enumerate(procs):
        if P["p"].returncode != 0:
            progress(f"!!! worker {k} exited with code {P['p'].returncode}; last log lines:\n{_tail(P['logp'])}")


def run_part(part, items, run_dir):
    """Runs every case of one Part (resumable): kernel-only items in the kernel, the rest in one worker per GPU
    (static split of the Part's combinations) or in the kernel."""
    sig = config_signature()
    slots, why = parallel_slots()
    t0 = time.time()
    ncases = sum(len(it["cases"]) for it in items)
    progress(f"==== PART {part}: {len(items)} combination(s), {ncases} cases; execution: {why} ({time.strftime('%H:%M:%S')})")
    kern = [it for it in items if it.get("kernel_only")]
    rest = [it for it in items if not it.get("kernel_only")]
    for it in kern:
        run_combo_cases9(it["combo"], it["cases"], run_dir, sig)
    if slots and rest:
        run_pool(rest, run_dir, slots, part)
    else:
        for it in rest:
            run_combo_cases9(it["combo"], it["cases"], run_dir, sig)
    progress(f"==== PART {part} finished in {fmt_time(time.time() - t0)}")


def collect(plan, run_dir):
    """All done records of the plan (current signature), the failure list and the missing list."""
    sig = config_signature()
    recs, failed, missing = {}, [], []
    for part in PARTS:
        for it in plan.get(part, []):
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
# SPEC9 analysis: works on whatever cases are on disk (SPEC9 10.1). Sections: HEAD, LIN, A, G, R, SUMMARY.
# Every verdict is printed with the numbers behind it; excluded units are listed with their reason.
# =============================================================================================================
def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _pearson_r2_slope(xs, ys):
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    sxx, syy, sxy = float(xm @ xm), float(ym @ ym), float(xm @ ym)
    r2 = sxy * sxy / (sxx * syy) if syy > 0 else float("nan")
    b = sxy / sxx
    return r2, b, float(y.mean() - b * x.mean())


def _ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    a, b = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else float("nan")


def _verdict(frac, used, rule_frac=None):
    rf = CFG["RULE_FRAC"] if rule_frac is None else rule_frac
    return "NO DATA" if not used else ("PASS" if frac >= rf else "FAIL")


def _pct(a, b):
    return 100.0 * a / b if b else float("nan")


R_BAD = ("DIVERGED", "JUMP-FAILED", "FAILED", "AJ-FAILED", "AJ-INCONSISTENT", "PRED-FAILED", "ERROR")


class Analysis9:
    SECTIONS = ("HEAD", "LIN", "A", "G", "R", "SUMMARY")

    def __init__(self, plan, run_dir, start_infos):
        self.plan, self.run_dir, self.si = plan, run_dir, start_infos or {}
        self.recs, self.failed, self.missing = collect(plan, run_dir)
        self.planned = {c["name"]: c for part in PARTS for it in plan.get(part, []) for c in it["cases"]}
        self.lines, self.sec = [], {k: [] for k in self.SECTIONS}
        self._pairs = {}
        self.supp = False
        self.H = []
        self.T = {}

    # ---------------------------------------------------------------- output
    def p(self, sec, s=""):
        self.sec[sec].append(s)
        self.lines.append(s)
        say(s)

    def rule(self, sec, d):
        d.setdefault("label", "SUPPLEMENTARY (hz on the step vector)" if self.supp else "PRIMARY")
        d["section"] = sec
        self.H.append(d)
        return d

    def _vec(self, name):
        try:
            return {k: v.to("cpu") for k, v in load_vecs(self.run_dir, name).items()}
        except Exception:
            return {}

    # ---------------------------------------------------------------- Part A / G even pairs
    def _f_of(self, c, rec):
        if rec is not None and rec.get("f_pred") is not None:
            return rec["f_pred"]
        info = self.si.get(c["combo"], {}).get("directions", {}).get(c["dir"], {})
        return predict_f9(c["L"], info.get("lam_ratio"))

    def pair(self, plus_name, need_conv):
        """Even-part pair of a Part A case (+ name): vL/vNP stats ('even'), G_L/G_NP stats ('geven'), dloss even part."""
        key = (plus_name, need_conv, self.supp)
        if key in self._pairs:
            return self._pairs[key]
        c = self.planned[plus_name]
        mn = plus_name[:-1] + "m"
        rp, rm = self.recs.get(plus_name), self.recs.get(mn)
        item = dict(name=plus_name[:-3], combo=c["combo"], learner=c["learner"], kind=c["L"]["kind"], dir=c["dir"],
                    size=c["size"], plus=rp, minus=rm, f=self._f_of(c, rp))
        if rp is None or rm is None:
            item["reason"] = "MISSING/FAILED"
        else:
            item["reason"] = usable9(rp, need_conv, self.supp) or usable9(rm, need_conv, self.supp)
            if not item["reason"]:
                vp, vm = self._vec(plus_name), self._vec(mn)
                if all(k in vp and k in vm for k in ("vL", "vNP", "GL", "GNP")):
                    item["even"] = even_stats(vp["vL"], vp["vNP"], vm["vL"], vm["vNP"])
                    item["geven"] = even_stats(vp["GL"], vp["GNP"], vm["GL"], vm["GNP"])
                    item["dloss_e"] = 0.5 * (rp["dloss_L"] + rm["dloss_L"])
                else:
                    item["reason"] = "no vectors"
        self._pairs[key] = item
        return item

    def A_units(self, group, sel, grp="A", dirs=None):
        """(combo, learner, dir) -> {"s1": pair, "s2": pair} for planned Part A cases of the (act, start) group."""
        out = {}
        for n, c in self.planned.items():
            if c["grp"] != grp or c["sign"] != 1 or not sel(c):
                continue
            a, s, _ = c["combo"].split("-")[:3]
            if (a, s) != group:
                continue
            if dirs is not None and c["dir"] not in dirs:
                continue
            nc = c["L"]["kind"] == "TM"
            out.setdefault((c["combo"], c["learner"], c["dir"]), {})[c["size"]] = self.pair(n, nc)
        return out

    @staticmethod
    def _ureason(u, sizes=("s1", "s2"), key="even"):
        for sz in sizes:
            x = u.get(sz)
            if x is None:
                return f"{sz}: NOT PLANNED"
            if x["reason"]:
                return f"{sz}: {x['reason']}"
            if key not in x:
                return f"{sz}: no vectors"
        return ""

    def groups_A(self):
        return sorted({tuple(c["combo"].split("-")[:2]) for c in self.planned.values() if c["grp"] == "A"})

    def rule_x9(self, sec, hname, group, sel, tol, key="even", conv_frac=None, special=None, what="rho"):
        """Extrapolated rule: x = (4 v(s2) - v(s1))/3 with cos, nr at s2; matches(x, cos(s2), nr(s2), f, tol) (or special)."""
        U = self.A_units(group, sel, dirs=CFG["A_DIRS"])
        used, excl = [], []
        for k, u in sorted(U.items()):
            why = self._ureason(u, key=key)
            if why:
                excl.append((k, why))
                continue
            e1, e2 = u["s1"][key], u["s2"][key]
            f = u["s1"]["f"]
            rx = (4.0 * e2["rho"] - e1["rho"]) / 3.0
            m = special(rx, e2, f) if special else matches(rx, e2["cos"], e2["nr"], f, tol)
            used.append(dict(combo=k[0], learner=k[1], dir=k[2], f=f, v_s1=e1["rho"], v_s2=e2["rho"], v_x=rx,
                             cos_s2=e2["cos"], nr_s2=e2["nr"], cos_s1=e1["cos"], match=bool(m)))
        nm = sum(x["match"] for x in used)
        frac = nm / len(used) if used else float("nan")
        verdict = _verdict(frac, used)
        extra = ""
        if conv_frac is not None:
            def conv(u):
                return all(u.get(sz) is not None and u[sz]["plus"] is not None and u[sz]["minus"] is not None
                           and u[sz]["plus"].get("learner_status") == "CONVERGED"
                           and u[sz]["minus"].get("learner_status") == "CONVERGED" for sz in ("s1", "s2"))
            nconv = sum(conv(u) for u in U.values())
            cf = nconv / len(U) if U else float("nan")
            extra = f"; converged (all four runs) {nconv}/{len(U)} planned units = {_pct(nconv, len(U)):.0f}% [rule >= {100 * conv_frac:.0f}%]"
            if used and not (cf >= conv_frac):
                verdict = "FAIL"
        rule_txt = (f"|{what}_x - 0.5| <= {CFG['HA5_TOL']} and cos(s2) >= {CFG['HA5_COS']}" if special
                    else f"matches({what}_x, cos(s2), nr(s2), f, tol {tol})")
        self.p(sec, f"{hname} {group[0]}/{group[1]}: {rule_txt} in {nm}/{len(used)} (learner, direction) units = "
                    f"{100 * frac:.1f}% [rule >= {100 * CFG['RULE_FRAC']:.0f}%]{extra} -> {verdict}")
        if excl:
            self.p(sec, f"    excluded and listed ({len(excl)}): " + "; ".join(f"{k[0]} {k[1]} {k[2]} [{w}]" for k, w in excl))
        self.p(sec, f"    combination        learner          dir     {what}(s1)  {what}(s2)     {what}_x          f   cos(s2)    nr(s2)  match")
        for x in used:
            self.p(sec, f"      {x['combo']:16s} {x['learner']:15s} {x['dir']:5s} {_fmt(x['v_s1'], 10)} {_fmt(x['v_s2'], 10)} "
                        f"{_fmt(x['v_x'], 10)} {_fmt(x['f'], 10)} {_fmt(x['cos_s2'], 9)} {_fmt(x['nr_s2'], 9)}  "
                        f"{'yes' if x['match'] else 'NO'}")
        return self.rule(sec, dict(rule=hname, group=list(group), matched=nm, used=len(used), frac=frac, verdict=verdict,
                                   excluded=[(" ".join(k), w) for k, w in excl], cases=used))

    # ---------------------------------------------------------------- linear control
    def section_LIN(self):
        S = "LIN"
        rs = [r for r in self.recs.values() if r["grp"] == "lin"]
        planned = [c for c in self.planned.values() if c["grp"] == "lin"]
        tol = CFG["H0_TOL"]

        def ok(r):
            return (_fin(r.get("vL_over_dL")) and r["vL_over_dL"] <= tol and _fin(r.get("vNP_over_dNP"))
                    and r["vNP_over_dNP"] <= tol and _fin(r.get("GL_over_dL")) and r["GL_over_dL"] <= tol)
        good = [r for r in rs if ok(r)]
        verdict = "PASS" if rs and len(good) == len(planned) else ("NO DATA" if not rs else "FAIL")
        mx = lambda k: max([r.get(k, float("nan")) for r in rs] or [float("nan")])
        self.p(S, "=" * 110)
        self.p(S, "LINEAR CONTROL (SPEC9 sec. 7)")
        self.p(S, f"H0 linear control (196 -> {CFG['N_CLASSES']} linear, init, seed 0; label, rand; s1; both signs; "
                  f"{', '.join(learner_name9(L_from_spec(x)) for x in CFG['LIN_LEARNERS'])}): {len(good)}/{len(planned)} runs "
                  f"with |v_L|/|theta_L-theta0|, |v_NP|/|theta_NP-theta0| and |G_L|/|F_test(theta_L)-F_test(theta0)| <= {tol} "
                  f"[rule: all] -> {verdict}   (max: {mx('vL_over_dL'):.2e}, {mx('vNP_over_dNP'):.2e}, {mx('GL_over_dL'):.2e}; "
                  f"learner status {sorted(set(str(r.get('learner_status')) for r in rs))})")
        for r in rs:
            if r not in good:
                self.p(S, f"      failing: {r['name']} {r.get('vL_over_dL')} {r.get('vNP_over_dNP')} {r.get('GL_over_dL')}")
        for c in planned:
            if c["name"] not in self.recs:
                self.p(S, f"      missing/failed: {c['name']}")
        self.rule(S, dict(rule="H0", group=["linear", "init"], matched=len(good), used=len(planned),
                          frac=len(good) / len(planned) if planned else float("nan"), verdict=verdict,
                          cases=[{k: r.get(k) for k in ("name", "vL_over_dL", "vNP_over_dNP", "GL_over_dL", "learner_status")}
                                 for r in rs]))

    # ---------------------------------------------------------------- Part A
    def section_A(self):
        S = "A"
        self.p(S, "=" * 110)
        self.p(S, "PART A - evaluation-point families (SPEC9 sec. 4). Extrapolated rho_x = (4 rho(s2) - rho(s1))/3, cos and nr at s2.")
        if CFG["MODE"] == "tiny":
            self.p(S, "MODE tiny: reduced grid (part of the mode definition): directions " + ", ".join(CFG["A_DIRS"])
                   + f"; AJ alpha {CFG['AJ_ALPHAS']}; SLa {CFG['SLA_SET']}; TJ {CFG['TJ_SET']}; TM {CFG['TM_SET']} on {CFG['TM_DIRS']}.")
        kinds = {"AJ": "HA1", "SLa": "HA2", "TJ": "HA3"}
        anchors = [learner_name9(L_from_spec(x)) for x in CFG["ANCHORS"]]
        ha5 = set(CFG["HA5_LEARNERS"])
        out = {}
        for g in self.groups_A():
            self.p(S, f"\n---- Part A {g[0]}/{g[1]} ----")
            for kd, hn in kinds.items():
                self.rule_x9(S, hn, g, lambda c, kd=kd: c["L"]["kind"] == kd, CFG["TOL_A"])
            self.rule_x9(S, "HA4", g, lambda c: c["L"]["kind"] == "TM", CFG["TOL_TM"], conv_frac=CFG["HA4_CONV_FRAC"])
            self.TM_inner_report(S, g)
            self.rule_x9(S, "HA5", g, lambda c: c["learner"] in ha5, None,
                         special=lambda rx, e2, f: abs(rx - 0.5) <= CFG["HA5_TOL"] and _fin(e2["cos"]) and e2["cos"] >= CFG["HA5_COS"])
            self.rule_HA6(S, g)
            self.rule_x9(S, "HA7", g, lambda c: c["learner"] in anchors, CFG["TOL_A"])
            if self.supp_differs(S, "A", g):
                self.p(S, f"-- SUPPLEMENTARY (not the pre-registered rules; A3.1: status from hz on the step vector): HA1-HA5, HA7 with the run status taken "
                          f"from hz evaluated on the step vector J_e^T a instead of theta - theta_b --")
                self.supp = True
                for kd, hn in kinds.items():
                    self.rule_x9(S, hn + "s", g, lambda c, kd=kd: c["L"]["kind"] == kd, CFG["TOL_A"])
                self.rule_x9(S, "HA4s", g, lambda c: c["L"]["kind"] == "TM", CFG["TOL_TM"], conv_frac=CFG["HA4_CONV_FRAC"])
                self.rule_x9(S, "HA5s", g, lambda c: c["learner"] in ha5, None,
                             special=lambda rx, e2, f: abs(rx - 0.5) <= CFG["HA5_TOL"] and _fin(e2["cos"]) and e2["cos"] >= CFG["HA5_COS"])
                self.rule_x9(S, "HA7s", g, lambda c: c["learner"] in anchors, CFG["TOL_A"])
                self.supp = False
            out[f"{g[0]}/{g[1]}"] = self.A_reports(S, g)
        self.T["A_reports"] = out

    def supp_differs(self, S, grp, g=None):
        """True when some run's SUPPLEMENTARY status differs from its literal status (then the supplementary rules are printed)."""
        rs = [r for r in self.recs.values() if r["grp"] == grp and (g is None or (r["act"], r["start"]) == tuple(g))]
        diff = [r["name"] for r in rs if "learner_status_supp" in r and r["learner_status_supp"] != r.get("learner_status")]
        if diff:
            self.p(S, f"    runs whose status (A3.1 rule) differs from the SUPPLEMENTARY status "
                      f"(hz from the step vector): {len(diff)} of {len(rs)}: " + ", ".join(diff[:40]) + (" ..." if len(diff) > 40 else ""))
        return bool(diff)

    def TM_inner_report(self, S, g):
        rs = [r for r in self.recs.values() if r["grp"] == "A" and (r["act"], r["start"]) == g and r["L"]["kind"] == "TM"]
        capped = [r for r in rs if r.get("learner_tm_inner_capped", 0)]
        self.p(S, f"    TM inner solves that hit the cap of {CFG['TM_INNER_CAP']} iterations (TM-INNER-FAILED, step taken with the last "
                  f"delta; recorded and listed): {len(capped)}/{len(rs)} runs have >= 1 such step")
        self.p(S, f"    A3.2 rounding-floor accepts (ii): total tm_inner_floor_accepts = "
                  f"{sum(int(r.get('learner_tm_inner_floor_accepts', 0)) for r in rs)}, max tm_floor_g_max = "
                  f"{max([r.get('learner_tm_floor_g_max', 0.0) for r in rs] or [0.0]):.3f} (<= TM_INNER_FP (1 + c)); total inner "
                  f"iterations {sum(int(r.get('learner_tm_inner_total', 0)) for r in rs)} over {sum(int(r.get('steps') or 0) for r in rs)} steps")
        for r in sorted(capped, key=lambda r: r["name"]):
            self.p(S, f"      TM-INNER-FAILED steps: {r['name']}: {r['learner_tm_inner_capped']}/{r['steps']} steps, worst |g|/tol "
                      f"{r['learner_tm_worst_g_over_tol']:.2f}, run status {r['learner_status']}")
        # SUPPLEMENTARY: HA4 restricted to units whose runs had no capped inner solve
        U = self.A_units(g, lambda c: c["L"]["kind"] == "TM", dirs=CFG["A_DIRS"])
        rows = []
        for k, u in sorted(U.items()):
            if self._ureason(u):
                continue
            runs = [u[sz][sg] for sz in ("s1", "s2") for sg in ("plus", "minus")]
            if any(r.get("learner_tm_inner_capped", 0) for r in runs):
                continue
            e1, e2 = u["s1"]["even"], u["s2"]["even"]
            rx = (4.0 * e2["rho"] - e1["rho"]) / 3.0
            rows.append(matches(rx, e2["cos"], e2["nr"], u["s1"]["f"], CFG["TOL_TM"]))
        self.p(S, f"    SUPPLEMENTARY (not a pre-registered rule): HA4 restricted to units with no TM-INNER-FAILED step: "
                  f"{sum(rows)}/{len(rows)} match")

    def rule_HA6(self, S, g):
        rs = [r for n, r in self.recs.items() if r["grp"] == "A" and (r["act"], r["start"]) == g
              and r["learner"] == "AJ(1)" and r["dir"] in CFG["A_DIRS"]]
        planned = [c for c in self.planned.values() if c["grp"] == "A" and c["learner"] == "AJ(1)"
                   and tuple(c["combo"].split("-")[:2]) == g and c["dir"] in CFG["A_DIRS"]]
        used, excl = [], []
        for r in rs:
            why = usable9(r) if usable9(r) not in ("NP-FAILED", "BELOW-RESOLUTION", "INVALID", "TH-JUMP-FAILED") else ""
            if why:
                excl.append((r["name"], why))
            elif r.get("ha6_np0_status") != "CONVERGED":
                excl.append((r["name"], f"NP0 {r.get('ha6_np0_status')}"))
            else:
                used.append(r)
        excl += [(c["name"], "MISSING/FAILED") for c in planned if c["name"] not in self.recs]
        ok = [r for r in used if _fin(r["ha6_rel"]) and r["ha6_rel"] <= CFG["HA6_REL"]]
        frac = len(ok) / len(used) if used else float("nan")
        verdict = _verdict(frac, used)
        vals = [r["ha6_rel"] for r in used]
        self.p(S, f"HA6 {g[0]}/{g[1]}: |theta_AJ(1) - theta_NP0| <= {CFG['HA6_REL']:g} |theta_NP0 - theta0| (NP0 = nearest_point started "
                  f"from theta0) in {len(ok)}/{len(used)} runs = {100 * frac:.1f}% [rule >= 90%] -> {verdict}   "
                  f"(max rel {max(vals) if vals else float('nan'):.2e}, median {_med(vals):.2e})")
        self.p(S, "    note: measure_ext starts its nearest-point solve at the learner's endpoint, which for AJ(1) already satisfies "
                  "the post-point condition, so rho = 1 for AJ(1) holds by construction; HA6 is the non-trivial check.")
        if excl:
            self.p(S, "    excluded and listed: " + "; ".join(f"{a} [{b}]" for a, b in excl))
        for r in used:
            if r not in ok:
                self.p(S, f"      failing: {r['name']} rel {r['ha6_rel']:.3e} (NP0 its {r.get('ha6_np0_it')})")
        return self.rule(S, dict(rule="HA6", group=list(g), matched=len(ok), used=len(used), frac=frac, verdict=verdict,
                                 excluded=excl, cases=[dict(name=r["name"], rel=r["ha6_rel"]) for r in used]))

    def A_reports(self, S, g):
        out = {}
        self.p(S, f"-- Part A {g[0]}/{g[1]} report (no verdicts) --")
        # literal single-size matches
        self.p(S, "  literal single-size matches (each size separately, even part, same matches() and tol):")
        lit = []
        for fam, tol in (("AJ", CFG["TOL_A"]), ("SLa", CFG["TOL_A"]), ("TJ", CFG["TOL_A"]), ("TM", CFG["TOL_TM"]), ("anchor", CFG["TOL_A"])):
            anchors = [learner_name9(L_from_spec(x)) for x in CFG["ANCHORS"]]
            sel = (lambda c: c["learner"] in anchors) if fam == "anchor" else (lambda c, fam=fam: c["L"]["kind"] == fam)
            U = self.A_units(g, sel, dirs=CFG["A_DIRS"])
            for sz in ("s1", "s2"):
                xs = [u[sz] for u in U.values() if sz in u and not u[sz]["reason"] and "even" in u[sz]]
                m = sum(matches(x["even"]["rho"], x["even"]["cos"], x["even"]["nr"], x["f"], tol) for x in xs)
                self.p(S, f"    {fam:6s} {sz}: {m}/{len(xs)} match")
                lit.append(dict(family=fam, size=sz, matched=m, used=len(xs)))
        out["literal"] = lit
        # order check
        self.p(S, "  order check: |rho(s1)-f| / |rho(s2)-f| (expected about 4)")
        self.p(S, "    combination        learner          dir    |rho(s1)-f|  |rho(s2)-f|    ratio       rho_x          f")
        U = self.A_units(g, lambda c: True, dirs=CFG["A_DIRS"])
        oc = []
        for k in sorted(U):
            u = U[k]
            why = self._ureason(u)
            if why:
                continue
            f = u["s1"]["f"]
            r1, r2 = u["s1"]["even"]["rho"], u["s2"]["even"]["rho"]
            a, b = abs(r1 - f), abs(r2 - f)
            ratio = a / b if b > 0 else float("inf")
            self.p(S, f"      {k[0]:16s} {k[1]:15s} {k[2]:5s} {_fmt(a, 12, 6)} {_fmt(b, 12, 6)} {_fmt(ratio, 8, 2)} "
                      f"{_fmt((4 * r2 - r1) / 3, 11)} {_fmt(f, 10)}")
            oc.append(dict(combo=k[0], learner=k[1], dir=k[2], err_s1=a, err_s2=b, ratio=ratio))
        out["order_check"] = oc
        # AJ outer iterations and dampings
        self.p(S, "  alpha-point jumps: outer iterations per AJ call and dampings, per learner (all sizes/signs/directions):")
        self.p(S, "    learner           runs  AJ calls  outer its/call (median, max)  dampings  max hz      max res   statuses"
                  "   | A3.1: max (hz - B)   jumps with hz > 1e-7 (original rule)")
        rs = [r for r in self.recs.values() if r["grp"] == "A" and (r["act"], r["start"]) == g
              and r["L"]["kind"] in ("AJ", "SLa", "TJ")]
        aj = []
        for ln in sorted({r["learner"] for r in rs}):
            xs = [r for r in rs if r["learner"] == ln]
            calls = sum(r.get("learner_aj_calls", 0) for r in xs)
            tot = sum(r.get("learner_aj_outer_total", 0) for r in xs)
            mx = max([r.get("learner_aj_outer_max", 0) for r in xs] or [0])
            dmp = sum(r.get("learner_aj_dampings", 0) for r in xs)
            hz = max([r.get("learner_aj_hz_max", 0.0) for r in xs] or [0.0])
            res = max([r.get("learner_aj_res_max", 0.0) for r in xs] or [0.0])
            stt = sorted({str(r.get("learner_status")) for r in xs})
            exc = max([r["learner_aj_hz_excess_max"] for r in xs if _fin(r.get("learner_aj_hz_excess_max"))] or [float("nan")])
            lit = sum(int(r.get("learner_aj_lit_flags", 0)) for r in xs)
            self.p(S, f"    {ln:16s} {len(xs):5d} {calls:9d}   {tot / max(calls, 1):8.2f} {mx:6d}              {dmp:8d}  {hz:.2e}  {res:.2e}  {stt}"
                      f"   | {exc:+.2e}            {lit}")
            aj.append(dict(learner=ln, runs=len(xs), calls=calls, mean_outer=tot / max(calls, 1), max_outer=mx, dampings=dmp,
                           hz_max=hz, res_max=res, statuses=stt, hz_excess_max=exc, lit_flags=lit))
        out["aj_iterations"] = aj
        # failed / excluded runs with reasons
        bad = [r for r in self.recs.values() if r["grp"] == "A" and (r["act"], r["start"]) == g and usable9(r)]
        self.p(S, f"  runs excluded from the rules (single runs; units containing them are excluded): {len(bad)}")
        for r in sorted(bad, key=lambda r: r["name"]):
            self.p(S, f"    {r['name']}: {usable9(r)} {r.get('learner_aj_why', '') or ''}")
        return out

    # ---------------------------------------------------------------- Part G
    def section_G(self):
        S = "G"
        self.p(S, "=" * 110)
        self.p(S, "PART G - predictions on unseen inputs (SPEC9 sec. 5); data_source=" + str(DATA_SOURCE)
               + ("  [SYNTHETIC held-out inputs: Gaussian, random labels, standardised with the training scalars]"
                  if DATA_SOURCE != "mnist" else "  [MNIST test split, 10 000 images]"))
        self.T["G2"], self.T["G2s"], self.T["G3"] = {}, {}, {}
        for g in self.groups_A():
            self.p(S, f"\n---- Part G {g[0]}/{g[1]} ----")
            self.rule_x9(S, "HG1", g, lambda c: True, CFG["TOL_G"], key="geven", what="gamma")
            self.rule_HG2(S, g)
            if any("learner_status_supp" in r and r["learner_status_supp"] != r.get("learner_status") for r in self.recs.values()
                   if r["grp"] == "A" and (r["act"], r["start"]) == g):
                self.p(S, "-- SUPPLEMENTARY (A3.1: status from hz on the step vector): HG1, HG2 with the run status from hz on the step vector --")
                self.supp = True
                self.rule_x9(S, "HG1s", g, lambda c: True, CFG["TOL_G"], key="geven", what="gamma")
                self.rule_HG2(S, g, name="HG2s")
                self.supp = False
        self.report_G3(S)
        self.report_G4(S)

    def G2_rows(self, combo, d, size="s2"):
        """Learners of (combo, dir) at the given size with a usable pair: (learner, f, dloss even part)."""
        rows = []
        for n, c in self.planned.items():
            if c["grp"] != "A" or c["combo"] != combo or c["dir"] != d or c["size"] != size or c["sign"] != 1:
                continue
            x = self.pair(n, c["L"]["kind"] == "TM")
            if not x["reason"] and "dloss_e" in x and _fin(x["f"]):
                rows.append((c["learner"], x["f"], x["dloss_e"]))
        return rows

    def rule_HG2(self, S, g, name="HG2"):
        combos_ = sorted({c["combo"] for c in self.planned.values() if c["grp"] == "A" and tuple(c["combo"].split("-")[:2]) == g})
        used, excl, rows_out = [], [], []
        self.p(S, f"{name} {g[0]}/{g[1]}: per (combination, direction) at s2: Pearson r^2 between the even part of "
                  f"dl_L = l(theta_L) - l(theta_TH) and f_L over the learners")
        self.p(S, "    combination        dir    learners        r^2      slope b (dl per unit f)   intercept")
        for cb in combos_:
            for d in CFG["A_DIRS"]:
                rows = self.G2_rows(cb, d)
                r2, b, a = _pearson_r2_slope([x[1] for x in rows], [x[2] for x in rows])
                self.T["G2" + ("s" if self.supp else "")][f"{cb}|{d}"] = dict(n=len(rows), r2=r2, b=b, a=a, rows=rows)
                if not _fin(r2):
                    excl.append((f"{cb} {d}", f"{len(rows)} usable learners (need >= 3)"))
                    continue
                m = r2 >= CFG["HG2_R2"]
                used.append(dict(combo=cb, dir=d, n=len(rows), r2=r2, b=b, match=m))
                self.p(S, f"      {cb:16s} {d:5s} {len(rows):5d}     {r2:10.6f}   {b:+.6e}          {a:+.3e}  {'yes' if m else 'NO'}")
        nm = sum(x["match"] for x in used)
        frac = nm / len(used) if used else float("nan")
        verdict = _verdict(frac, used)
        self.p(S, f"{name} {g[0]}/{g[1]}: r^2 >= {CFG['HG2_R2']} in {nm}/{len(used)} (combination, direction) pairs = {100 * frac:.1f}% "
                  f"[rule >= 90%] -> {verdict}")
        if excl:
            self.p(S, "    excluded and listed: " + "; ".join(f"{a} [{b}]" for a, b in excl))
        self.rule(S, dict(rule=name, group=list(g), matched=nm, used=len(used), frac=frac, verdict=verdict, excluded=excl,
                          cases=used))

    def report_G3(self, S):
        self.p(S, "\n---- G3 (OPEN QUESTION - no prediction, two-sided): does alignment help? ----")
        self.p(S, "no prediction was registered for the sign of b")
        self.p(S, "b = slope of the even part of dl_L = l(theta_L) - l(theta_TH) against f at s2 (b < 0: moving toward the nearest "
                  "solution LOWERS the test loss). E+ = [l(NP) - l(TH)] / [l(theta0) - l(TH)] from the + sign runs (median and range "
                  "over the learners), and E+ / (s/S_dir).")
        combos_ = sorted({c["combo"] for c in self.planned.values() if c["grp"] == "A"})
        nneg = {}
        self.p(S, "    combination        dir     n      b              sign   size  E+ median     E+ range                  E+/(s/S) median")
        for cb in combos_:
            for d in ("true", "label", "rand"):
                if not any(c["combo"] == cb and c["dir"] == d for c in self.planned.values() if c["grp"] == "A"):
                    continue
                rows = self.G2_rows(cb, d)
                r2, b, a = _pearson_r2_slope([x[1] for x in rows], [x[2] for x in rows])
                if _fin(b):
                    nneg.setdefault(d, [0, 0])
                    nneg[d][1] += 1
                    nneg[d][0] += int(b < 0)
                for sz in ("s1", "s2"):
                    Es, Er = [], []
                    for n, c in self.planned.items():
                        if c["grp"] != "A" or c["combo"] != cb or c["dir"] != d or c["size"] != sz or c["sign"] != 1:
                            continue
                        r = self.recs.get(n)
                        if r is None or usable9(r, c["L"]["kind"] == "TM"):
                            continue
                        den = r["loss_0"] - r["loss_TH"]
                        if abs(den) < 1e-12:
                            continue
                        E = (r["loss_NP"] - r["loss_TH"]) / den
                        Es.append(E)
                        Er.append(E / (r["s"] / r["S_dir"]))
                    rng = f"[{min(Es):+.3e}, {max(Es):+.3e}]" if Es else "undefined"
                    self.p(S, f"      {cb:16s} {d:5s} {len(rows):4d}  {_fmt(b, 13, 4)}  {('-' if b < 0 else '+') if _fin(b) else '?':>4s}   "
                              f"{sz}   {_fmt(_med(Es), 11, 4)}  {rng:24s}  {_fmt(_med(Er), 11, 4)}")
                    self.T["G3"][f"{cb}|{d}|{sz}"] = dict(b=b, r2=r2, E_med=_med(Es), E_min=min(Es) if Es else None,
                                                          E_max=max(Es) if Es else None, E_per_rel_med=_med(Er), nE=len(Es))
        for d, (k, n) in sorted(nneg.items()):
            self.p(S, f"    (iii) direction {d}: b < 0 in {k} of {n} combinations")
        self.T["G3_count_b_negative"] = nneg

    def report_G4(self, S):
        self.p(S, "\n---- G4 (report only): large moves toward the true labels, t = F0 + tau (Y_true - F0), sign + ----")
        if CFG["MODE"] == "tiny":
            self.p(S, f"MODE tiny: G4 reduced to {CFG['G4_COMBOS']} (act, seed, n), tau in {CFG['G4_TAUS']}; the n = N_TRAIN "
                      f"rule for the SPEC8 measurement applies to n = {CFG['N_TRAIN']}.")
        rs = [r for r in self.recs.values() if r["grp"] == "G4"]
        out = {}
        for cb in sorted({c["combo"] for c in self.planned.values() if c["grp"] == "G4"}):
            for tau in CFG["G4_TAUS"]:
                xs = [r for r in rs if r["combo"] == cb and abs(r["tau"] - tau) < 1e-12]
                miss = [c["name"] for c in self.planned.values() if c["grp"] == "G4" and c["combo"] == cb
                        and abs(c["frac"] - tau) < 1e-12 and c["name"] not in self.recs]
                self.p(S, f"  {cb} tau={tau:g}:")
                self.p(S, "      learner        f_nom    status        train res    test loss   test acc   |th_L-th0|    lambda1(K_L)  trK(K_L)     rho(NP)   cos(NP)  NP status")
                use = []
                for r in sorted(xs, key=lambda r: r["f_nominal"]):
                    good = r.get("learner_status") not in ("DIVERGED", "AJ-FAILED", "AJ-INCONSISTENT") and _fin(r.get("train_res")) \
                        and r["train_res"] <= CFG["G4_RES_MAX"]
                    if good:
                        use.append(r)
                    self.p(S, f"      {r['learner']:13s} {r['f_nominal']:+.4f}  {str(r.get('learner_status')):13s} {_fmt(r.get('train_res'), 10)} "
                              f"{_fmt(r.get('loss_L'), 12, 6)} {_fmt(r.get('acc_L'), 9)} {_fmt(r.get('dist'), 12)} {_fmt(r.get('lam1_L'), 13)} "
                              f"{_fmt(r.get('trK_L'), 11)} {_fmt(r.get('np_rho'), 9)} {_fmt(r.get('np_cos'), 9)}  {r.get('np_status', '-')}"
                              f"{'' if good else '   <- excluded from the ranking (residual > %g or failed)' % CFG['G4_RES_MAX']}")
                for m in miss:
                    self.p(S, f"      MISSING/FAILED: {m}")
                rho_s = spearman([r["f_nominal"] for r in use], [r["loss_L"] for r in use])
                l0 = xs[0]["loss_0"] if xs else float("nan")
                self.p(S, f"      Spearman(f_nominal, test loss) over {len(use)} usable learners = {_fmt(rho_s, 8)}   (test loss at theta0 {_fmt(l0, 10, 6)})")
                out[f"{cb}|{tau:g}"] = dict(spearman=rho_s, n=len(use), rows=[{k: r.get(k) for k in (
                    "learner", "f_nominal", "learner_status", "train_res", "loss_L", "acc_L", "dist", "lam1_L", "trK_L", "np_rho",
                    "np_cos", "np_status")} for r in xs])
                if abs(tau - 1.0) < 1e-12:
                    base = "-".join(cb.split("-")[:3])
                    b = self.T["G2"].get(f"{base}|true", {}).get("b")
                    if b is None:
                        rows = self.G2_rows(base, "true")
                        _, b, _ = _pearson_r2_slope([x[1] for x in rows], [x[2] for x in rows])
                    self.p(S, f"      stated expectation (not a rule): sign(Spearman at tau=1) = sign(b of G3, direction true, {base}): "
                              f"Spearman {_fmt(rho_s, 8)}, b {_fmt(b, 12)} -> "
                              + ("holds" if _fin(rho_s) and _fin(b) and (rho_s > 0) == (b > 0) else
                                 ("undetermined" if not (_fin(rho_s) and _fin(b)) else "does NOT hold")))
        self.T["G4"] = out

    # ---------------------------------------------------------------- Part R
    def R_cases(self):
        return [r for r in self.recs.values() if r["grp"] == "R"]

    def R_pairs(self, act=None, primary=True):
        """(case record, learner record, usable reason) for Part R learners."""
        P = set(CFG["R_PRIMARY"])
        out = []
        for r in self.R_cases():
            if act is not None and r["act"] != act:
                continue
            for ln, lr in (r.get("learners") or {}).items():
                if primary and ln not in P:
                    continue
                if not primary and ln in P:
                    continue
                out.append((r, lr, self.R_reason(r, lr)))
        return out

    def R_reason(self, r, lr):
        if r.get("ref_status") != "OK":
            return "REF-FAILED"
        stt = lr.get("learner_status_supp", lr.get("learner_status")) if self.supp else lr.get("learner_status")
        if stt in R_BAD:
            return str(stt)
        if lr.get("out_of_range", 0):
            return "OUT-OF-RANGE"
        if not (_fin(r.get("nvPP")) and r["nvPP"] >= CFG["VPP_MIN_REL"] * r.get("dref1", float("inf"))):
            return "v_PP below 1e-9 |theta_ref(1) - theta0|"
        return ""

    def hr1_ok(self, lr):
        return (_fin(lr.get("cosF")) and lr["cosF"] >= CFG["HR1_COS"] and _fin(lr.get("nrF"))
                and CFG["HR1_NR"][0] <= lr["nrF"] <= CFG["HR1_NR"][1])

    def fmatch(self, lr, tol):
        f = lr.get("f_smooth")
        return _fin(lr.get("rho_PP")) and abs(lr["rho_PP"] - f) <= tol * max(1.0, abs(f))

    def section_R(self):
        S = "R"
        self.p(S, "=" * 110)
        self.p(S, "PART R - ReLU and the general (reference-path) form of the law (SPEC9 sec. 6). No nearest-point solve (except R-e).")
        self.p(S, f"Every sign is a separate case (no even part, no extrapolation). Reference grid G_REF = {CFG['G_REF']} cells on [0, 2]"
               + (" (MODE tiny: 64 cells, sizes and directions reduced as defined for the mode)" if CFG["MODE"] == "tiny" else ""))
        acts = sorted({r["act"] for r in self.R_cases()} | {c["combo"].split("-")[0] for c in self.planned.values() if c["grp"] == "R"})
        miss = [c["name"] for c in self.planned.values() if c["grp"] == "R" and c["name"] not in self.recs]
        if miss:
            self.p(S, f"!!! Part R cases without a result ({len(miss)}): " + ", ".join(miss[:30]) + (" ..." if len(miss) > 30 else ""))
        reff = [r["name"] for r in self.R_cases() if r.get("ref_status") != "OK"]
        if reff:
            self.p(S, "!!! REF-FAILED cases (excluded, listed): " + ", ".join(reff))
        self.R_rules(S, acts)
        dif = [(r["name"], ln) for r in self.R_cases() for ln, lr in (r.get("learners") or {}).items()
               if "learner_status_supp" in lr and lr["learner_status_supp"] != lr.get("learner_status")]
        if dif:
            self.p(S, f"\n-- SUPPLEMENTARY (A3.1: status from hz on the step vector): HR1-HR5 with the status from hz on the step vector; differing pairs: {dif} --")
            self.supp = True
            self.R_rules(S, acts, suffix="s")
            self.supp = False
        self.R_reports(S)

    def R_rules(self, S, acts, suffix=""):
        for a in acts:
            role = "the test" if a == "relu" else "smooth control"
            self.p(S, f"\n---- Part R {a} ({role}) ----")
            pairs = self.R_pairs(a)
            us = [(r, lr) for r, lr, why in pairs if not why]
            ex = [(r, lr, why) for r, lr, why in pairs if why]
            if ex:
                self.p(S, f"    unusable (case, learner) pairs ({len(ex)}): " + "; ".join(f"{r['name']} {lr.get('learner')} [{w}]" for r, lr, w in ex))
            # HR1
            ok = [x for x in us if self.hr1_ok(x[1])]
            fr = len(ok) / len(us) if us else float("nan")
            v = _verdict(fr, us)
            self.p(S, f"HR1{suffix} {a}: cosF >= {CFG['HR1_COS']} and nrF in {list(CFG['HR1_NR'])} in {len(ok)}/{len(us)} usable (case, learner) "
                      f"pairs = {100 * fr:.1f}% [rule >= 90%] -> {v}   (min cosF {_fmt(min([x[1]['cosF'] for x in us] or [float('nan')]), 9, 6)}, "
                      f"nrF range {_fmt(min([x[1]['nrF'] for x in us] or [float('nan')]))} .. {_fmt(max([x[1]['nrF'] for x in us] or [float('nan')]))})")
            per = {}
            for r, lr in us:
                per.setdefault(r["frac"], [0, 0])
                per[r["frac"]][1] += 1
                per[r["frac"]][0] += int(self.hr1_ok(lr))
            self.p(S, "    HR1 pass fraction per size: " + "; ".join(f"s/S0={k:g}: {a_}/{b_}" for k, (a_, b_) in sorted(per.items())))
            for r, lr in us:
                if not self.hr1_ok(lr):
                    self.p(S, f"      HR1 failing: {r['name']} {lr['learner']} cosF {lr['cosF']:.6f} nrF {lr['nrF']:.4f} status {lr['learner_status']}")
            self.rule(S, dict(rule="HR1" + suffix, group=[a], matched=len(ok), used=len(us), frac=fr, verdict=v,
                              per_size={f"{k:g}": x for k, x in per.items()}))
            # HR2
            ok2 = [x for x in us if _fin(x[1].get("rho_omega")) and abs(x[1]["rho_PP"] - x[1]["rho_omega"])
                   <= CFG["HR2_TOL"] * max(1.0, abs(x[1]["rho_PP"]))]
            fr2 = len(ok2) / len(us) if us else float("nan")
            v2 = _verdict(fr2, us)
            dev = [abs(x[1]["rho_PP"] - x[1]["rho_omega"]) for x in us]
            self.p(S, f"HR2{suffix} {a}: |rho_PP - rho_omega| <= {CFG['HR2_TOL']} max(1, |rho_PP|) in {len(ok2)}/{len(us)} = {100 * fr2:.1f}% "
                      f"[rule >= 90%] -> {v2}   (max |rho_PP - rho_omega| {_fmt(max(dev or [float('nan')]))}, median {_fmt(_med(dev))})")
            bp = [x[1].get("bperp_max", float("nan")) for x in us]
            rp_pass = [abs(x[1].get("rho_perp", float("nan"))) for x in ok2]
            fails2 = [x for x in us if x not in ok2]
            self.p(S, f"    HR2 note (A3.4, written before the GPU run): P-omega keeps only the component of each b_k along d. Usable pairs: "
                      f"median / max bperp_max = {_fmt(_med(bp)).strip()} / {_fmt(max([v for v in bp if _fin(v)] or [float('nan')])).strip()}; "
                      f"HR2-passing pairs: median |rho_perp| = {_fmt(_med(rp_pass)).strip()}; HR2-failing pairs: median |rho_perp| = "
                      f"{_fmt(_med([abs(x[1].get('rho_perp', float('nan'))) for x in fails2])).strip()}, median bperp_max = "
                      f"{_fmt(_med([x[1].get('bperp_max', float('nan')) for x in fails2])).strip()}. A FAIL of HR2 together with a PASS of "
                      f"HR1 means that the scalar reduction is invalid there, not the general law.")
            for r, lr in us:
                if (r, lr) not in ok2:
                    self.p(S, f"      HR2 failing: {r['name']} {lr['learner']} rho_PP {lr['rho_PP']:+.4f} rho_omega {lr['rho_omega']:+.4f}")
            self.rule(S, dict(rule="HR2" + suffix, group=[a], matched=len(ok2), used=len(us), frac=fr2, verdict=v2))
            conv = [(r, lr) for r, lr in us if (lr.get("learner_status_supp", lr.get("learner_status")) if self.supp
                                                else lr.get("learner_status")) == "CONVERGED"]
            if a != "relu":
                self.rule_HR3(S, a, conv, suffix)
            else:
                self.rule_HR45(S, conv, suffix)

    def rule_HR3(self, S, a, conv, suffix=""):
        smax = CFG["HR3_SIZE_MAX"] * (1 + 1e-12)
        cases = [r for r in self.R_cases() if r["act"] == a and r["frac"] <= smax and r.get("ref_status") == "OK"
                 and _fin(r.get("nvPP")) and r["nvPP"] >= CFG["VPP_MIN_REL"] * r.get("dref1", float("inf"))]
        dev = [max(abs(r["omega"][k] - k / 16.0) for k in range(1, 17)) for r in cases]
        okc = sum(x <= CFG["HR3_OMEGA_TOL"] for x in dev)
        frc = okc / len(cases) if cases else float("nan")
        ps = [(r, lr) for r, lr in conv if r["frac"] <= smax]
        okp = [x for x in ps if self.fmatch(x[1], CFG["HR_F_TOL"])]
        frp = len(okp) / len(ps) if ps else float("nan")
        v = "NO DATA" if not cases or not ps else ("PASS" if frc >= CFG["RULE_FRAC"] and frp >= CFG["RULE_FRAC"] else "FAIL")
        self.p(S, f"HR3{suffix} {a} (sizes <= {CFG['HR3_SIZE_MAX']:g}): max_tau in 1/16..1 |omega(tau) - tau| <= {CFG['HR3_OMEGA_TOL']} in {okc}/{len(cases)} "
                  f"cases = {100 * frc:.1f}% (max {_fmt(max(dev or [float('nan')]))}); |rho_PP - f| <= {CFG['HR_F_TOL']} max(1,|f|) in "
                  f"{len(okp)}/{len(ps)} CONVERGED (case, learner) pairs = {100 * frp:.1f}% [both >= 90%] -> {v}")
        for r, lr in ps:
            if (r, lr) not in okp:
                self.p(S, f"      HR3 failing pair: {r['name']} {lr['learner']} rho_PP {lr['rho_PP']:+.4f} f {lr['f_smooth']:+.4f}")
        self.rule(S, dict(rule="HR3" + suffix, group=[a], matched=[okc, len(okp)], used=[len(cases), len(ps)], frac=[frc, frp], verdict=v))

    def rule_HR45(self, S, conv, suffix=""):
        nos = [(r, lr) for r, lr in conv if not lr.get("ref_switch_0max1tau")]
        ok4 = [x for x in nos if self.fmatch(x[1], CFG["HR_F_TOL"])]
        if len(nos) < CFG["HR4_MIN_PAIRS"]:
            v4, fr4 = "NO DATA", float("nan")
            self.p(S, f"HR4{suffix} relu (no switch on [0, max(1, max tau_k)]): only {len(nos)} such CONVERGED pairs (< {CFG['HR4_MIN_PAIRS']}) -> NO DATA"
                      f"   ({len(ok4)}/{len(nos)} would match)")
        else:
            fr4 = len(ok4) / len(nos)
            v4 = _verdict(fr4, nos)
            self.p(S, f"HR4{suffix} relu (no switch on [0, max(1, max tau_k)]): |rho_PP - f| <= {CFG['HR_F_TOL']} max(1,|f|) in {len(ok4)}/{len(nos)} = "
                      f"{100 * fr4:.1f}% [rule >= 90%] -> {v4}")
        for r, lr in nos:
            if (r, lr) not in ok4:
                self.p(S, f"      HR4 failing: {r['name']} {lr['learner']} rho_PP {lr['rho_PP']:+.4f} f {lr['f_smooth']:+.4f}")
        self.rule(S, dict(rule="HR4" + suffix, group=["relu"], matched=len(ok4), used=len(nos), frac=fr4, verdict=v4))
        sw = [(r, lr) for r, lr in conv if lr.get("ref_switch_01")]
        miss = [x for x in sw if _fin(x[1].get("rho_PP")) and abs(x[1]["rho_PP"] - x[1]["f_smooth"]) > CFG["HR5_MISS"]]
        h1 = [x for x in sw if self.hr1_ok(x[1])]
        fm = len(miss) / len(sw) if sw else float("nan")
        fh = len(h1) / len(sw) if sw else float("nan")
        v5 = "NO DATA" if not sw else ("PASS" if fm >= CFG["HR5_FRAC"] and fh >= CFG["RULE_FRAC"] else "FAIL")
        self.p(S, f"HR5{suffix} relu (>= 1 switch on [0, 1]): smooth form misses (|rho_PP - f| > {CFG['HR5_MISS']}) in {len(miss)}/{len(sw)} = "
                  f"{100 * fm:.1f}% [rule >= {100 * CFG['HR5_FRAC']:.0f}%] AND HR1 condition in {len(h1)}/{len(sw)} = {100 * fh:.1f}% "
                  f"[rule >= 90%] -> {v5}")
        self.rule(S, dict(rule="HR5" + suffix, group=["relu"], matched=[len(miss), len(h1)], used=len(sw), frac=[fm, fh], verdict=v5))

    def R_reports(self, S):
        cases = sorted(self.R_cases(), key=lambda r: (r["act"], r["combo"], r["frac"], r["dir"], -r["sign"]))
        out = {}
        # R-a
        self.p(S, "\nR-a omega profiles on [0, 1] (17 values, tau = 0, 1/16, ..., 1), n_sw(1), switching cells on [0,1]/[0,2]:")
        ra = []
        for r in cases:
            if r.get("ref_status") != "OK":
                continue
            om = r["omega"][:17]
            self.p(S, f"    {r['name']:48s} " + " ".join(f"{x:+.3f}" for x in om)
                   + f" | n_sw(1)={r['nsw_1']} cells {r['cells_switch_01']}/{r['cells_switch_02']}")
            G = r["G_REF"]
            ch = r["cell_change"]
            tv, tvs = 0.0, 0.0
            for k in range(16):
                dv = abs(om[k + 1] - om[k])
                lo, hi = k * G // 32, (k + 1) * G // 32
                tv += dv
                if any(ch[lo:hi]):
                    tvs += dv
            ra.append(dict(name=r["name"], act=r["act"], frac=r["frac"], omega=om, omega_perp=r["omega_perp"][:17],
                           maxdev=max(abs(om[k] - k / 16.0) for k in range(17)), maxperp=max(r["omega_perp"][:17]),
                           nsw1=r["nsw_1"], tv=tv, tv_switch=tvs))
        self.p(S, "  R-a summary per (act, size): median over cases of max|omega - tau| on [0,1], of max omega_perp on [0,1], of n_sw(1);"
                  " share of the total variation of omega on [0,1] in 1/16-intervals containing a switching cell (pooled; median per case)")
        self.p(S, "    act    s/S0       cases  med max|om-tau|  med max om_perp  med n_sw(1)  TV share pooled  TV share median")
        summ = []
        for a, fr in sorted({(x["act"], x["frac"]) for x in ra}):
            xs = [x for x in ra if x["act"] == a and x["frac"] == fr]
            tvp = sum(x["tv_switch"] for x in xs) / max(sum(x["tv"] for x in xs), 1e-300)
            tvm = _med([x["tv_switch"] / x["tv"] for x in xs if x["tv"] > 0])
            self.p(S, f"    {a:6s} {fr:<9g} {len(xs):5d}  {_fmt(_med([x['maxdev'] for x in xs]), 14)}  {_fmt(_med([x['maxperp'] for x in xs]), 14)}"
                      f"  {_fmt(_med([x['nsw1'] for x in xs]), 11, 1)}  {_fmt(tvp, 14)}  {_fmt(tvm, 14)}")
            summ.append(dict(act=a, frac=fr, cases=len(xs), med_maxdev=_med([x['maxdev'] for x in xs]),
                             med_maxperp=_med([x['maxperp'] for x in xs]), med_nsw1=_med([x['nsw1'] for x in xs]),
                             tv_share_pooled=tvp, tv_share_median=tvm))
        out["R_a"] = dict(cases=ra, summary=summ)
        # R-b
        self.p(S, "\nR-b homogenisation per (act, size): medians over cases of rho_PP (CONVERGED, usable) and of |rho_PP - f| over P")
        self.p(S, "    act    s/S0       rho_PP SL(32)  [0.4844]   rho_PP DJ(0.5)  [0.3333]   med |rho_PP - f| over P   pairs")
        pairs = [(r, lr) for r, lr, why in self.R_pairs() if not why and lr.get("learner_status") == "CONVERGED"]
        rb = []
        for a, fr in sorted({(r["act"], r["frac"]) for r, _ in pairs}):
            xs = [(r, lr) for r, lr in pairs if r["act"] == a and r["frac"] == fr]
            s32 = _med([lr["rho_PP"] for r, lr in xs if lr["learner"] == "SL(32)"])
            dj5 = _med([lr["rho_PP"] for r, lr in xs if lr["learner"] == "DJ(0.5)"])
            dv = _med([abs(lr["rho_PP"] - lr["f_smooth"]) for r, lr in xs])
            self.p(S, f"    {a:6s} {fr:<9g} {_fmt(s32, 14)}            {_fmt(dj5, 14)}            {_fmt(dv, 14)}        {len(xs)}")
            rb.append(dict(act=a, frac=fr, rho_SL32=s32, rho_DJ05=dj5, med_dev=dv, n=len(xs)))
        self.p(S, "    stated expectation (not a rule): for relu the deviation from the smooth fractions first rises with size (first "
                  "switches) and then falls again when switches become many.")
        out["R_b"] = rb
        # R-c
        self.p(S, "\nR-c overshooting learners DJ(1.5), DJ(1.9) (not in the primary rules):")
        self.p(S, "    case                                             learner      cosF      nrF       rho_PP   rho_field        f  status   OOR")
        rc = []
        for r, lr, why in self.R_pairs(primary=False):
            self.p(S, f"    {r['name']:48s} {lr.get('learner', '?'):9s} {_fmt(lr.get('cosF'), 9, 5)} {_fmt(lr.get('nrF'), 8)} {_fmt(lr.get('rho_PP'), 9)} "
                      f"{_fmt(lr.get('rho_field'), 10)} {_fmt(lr.get('f_smooth'), 8)}  {lr.get('learner_status')}  {lr.get('out_of_range')}"
                      f"{'  [' + why + ']' if why else ''}")
            rc.append(dict(name=r["name"], act=r["act"], frac=r["frac"], **{k: lr.get(k) for k in (
                "learner", "cosF", "nrF", "rho_PP", "rho_field", "f_smooth", "learner_status", "out_of_range")}))
        out["R_c"] = rc
        # R-d
        self.p(S, "\nR-d accuracy of the general law versus the number of switches n_sw(1) (usable primary pairs):")
        self.p(S, "    act    n_sw(1) bin   pairs   median cosF    min cosF   median nrF   HR1-condition share")
        bins = [("0", 0, 0), ("1-2", 1, 2), ("3-10", 3, 10), ("11-100", 11, 100), (">100", 101, 10 ** 12)]
        rd = []
        for a in sorted({r["act"] for r in self.R_cases()}):
            us = [(r, lr) for r, lr, why in self.R_pairs(a) if not why]
            for bn, lo, hi in bins:
                xs = [(r, lr) for r, lr in us if lo <= r["nsw_1"] <= hi]
                if not xs:
                    continue
                cs = [lr["cosF"] for r, lr in xs]
                sh = sum(self.hr1_ok(lr) for r, lr in xs) / len(xs)
                self.p(S, f"    {a:6s} {bn:12s} {len(xs):6d}   {_fmt(_med(cs), 11, 6)}  {_fmt(min(cs), 10, 6)}  {_fmt(_med([lr['nrF'] for r, lr in xs]), 10)}   {sh:.3f}")
                rd.append(dict(act=a, bin=bn, n=len(xs), med_cosF=_med(cs), min_cosF=min(cs),
                               med_nrF=_med([lr['nrF'] for r, lr in xs]), hr1_share=sh))
        out["R_d"] = rd
        # R-e
        self.p(S, f"\nR-e v_PP against the SPEC8 nearest point (smooth controls, sizes <= {CFG['R_E_SIZE_MAX']:g}; one NP solve per case, from theta0):")
        re_ = []
        for r in cases:
            if "re_cos" not in r:
                continue
            self.p(S, f"    {r['name']:48s} cos(v_PP, v_NP) {r['re_cos']:.6f}   |v_PP|/|v_NP| {r['re_ratio']:.5f}   NP {r['re_np_status']} ({r['re_np_it']} its)")
            re_.append(dict(name=r["name"], cos=r["re_cos"], ratio=r["re_ratio"], np_status=r["re_np_status"]))
        out["R_e"] = re_
        # R-f (A3.4b): learner status counts per (act, size), outer iterations and dampings of the AJ-type learners
        self.p(S, "\nR-f status table (A3.4b, report only): per activation and size, learner status counts over the cases; for AJ-type "
                  "learners median/max over runs of the max outer iterations per AJ call and of the dampings per run")
        cats = ["CONVERGED", "UNCONVERGED", "AJ-FAILED", "AJ-INCONSISTENT", "JUMP-FAILED", "DIVERGED", "PRED-FAILED", "other"]
        self.p(S, "    act    s/S0       learner     " + " ".join(f"{c[:9]:>9s}" for c in cats)
               + "   outer its med/max   dampings med/max")
        rf = []
        lnames = [learner_name9(L_from_spec(x)) for x in CFG["R_LEARNERS"]]
        for a, fr in sorted({(r["act"], r["frac"]) for r in cases if r.get("ref_status") == "OK"}):
            xs = [r for r in cases if r["act"] == a and r["frac"] == fr and r.get("ref_status") == "OK"]
            for ln in lnames:
                lrs = [r["learners"][ln] for r in xs if ln in (r.get("learners") or {})]
                cnt = {c: 0 for c in cats}
                for lr in lrs:
                    stt = str(lr.get("learner_status"))
                    cnt[stt if stt in cnt else "other"] += 1
                row = dict(act=a, frac=fr, learner=ln, n=len(lrs), counts=cnt)
                extra = ""
                if ln.startswith(("AJ(", "SLa(")):
                    om = [lr.get("learner_aj_outer_max") for lr in lrs if lr.get("learner_aj_outer_max") is not None]
                    dm = [lr.get("learner_aj_dampings") for lr in lrs if lr.get("learner_aj_dampings") is not None]
                    row.update(outer_med=_med(om), outer_max=max(om) if om else None, damp_med=_med(dm),
                               damp_max=max(dm) if dm else None)
                    extra = (f"   {_fmt(_med(om), 6, 1)} / {str(max(om) if om else '-'):>4s}       "
                             f"{_fmt(_med(dm), 6, 1)} / {str(max(dm) if dm else '-'):>4s}")
                self.p(S, f"    {a:6s} {fr:<9g} {ln:10s} " + " ".join(f"{cnt[c]:9d}" for c in cats) + extra)
                rf.append(row)
        out["R_f"] = rf
        # per-case learner table
        self.p(S, "\nR per-case learner table (rho_PP, P-field rho_field / cosF / nrF, P-omega rho_omega, f, steps, max tau, n_sw at the endpoint, switch on [0,1]/[0,max tau], cost J, A3.4: bperp_max, rho_perp):")
        for r in cases:
            for ln, lr in (r.get("learners") or {}).items():
                self.p(S, f"    {r['name']:44s} {ln:10s} {_fmt(lr.get('rho_PP'), 8)} | {_fmt(lr.get('rho_field'), 8)} {_fmt(lr.get('cosF'), 9, 6)} "
                          f"{_fmt(lr.get('nrF'), 7)} | {_fmt(lr.get('rho_omega'), 8)} | f {_fmt(lr.get('f_smooth'), 7)} | {str(lr.get('learner_status'))[:11]:11s} "
                          f"{lr.get('steps')} {_fmt(lr.get('max_tau'), 6, 3)} {lr.get('nsw_end')} {int(bool(lr.get('ref_switch_01')))}/{int(bool(lr.get('ref_switch_0maxtau')))} "
                          f"{lr.get('n_J')} | {_fmt(lr.get('bperp_max'), 9)} {_fmt(lr.get('rho_perp'), 9)}")
        self.T["R_reports"] = out

    # ---------------------------------------------------------------- run
    def run(self):
        H = "HEAD"
        self.p(H, "=" * 110)
        self.p(H, f"SPEC9 ANALYSIS (data_source={DATA_SOURCE}; MODE={CFG['MODE']}; {len(self.recs)} case records on disk, "
                  f"{len(self.failed)} failed, {len(self.missing)} without a result)")
        if DATA_SOURCE != "mnist":
            self.p(H, "!!! data_source = synthetic: N(0,1) training inputs with random labels; Part G held-out inputs are synthetic too !!!")
        if self.failed:
            self.p(H, "!!! FAILED CASES (traceback in part9_run/failed/; retried when the notebook is run again):")
            for f in self.failed:
                self.p(H, f"!!!   {f['name']}: {f['error'].strip().splitlines()[-1]}")
        if self.missing:
            self.p(H, f"!!! {len(self.missing)} planned cases have no result yet (run interrupted or still to come): "
                      + ", ".join(self.missing[:12]) + (" ..." if len(self.missing) > 12 else ""))
        self.section_LIN()
        self.section_A()
        self.section_G()
        self.section_R()
        S = "SUMMARY"
        self.p(S, "\n" + "=" * 110 + f"\nVERDICT SUMMARY (data_source={DATA_SOURCE}; MODE={CFG['MODE']})")
        for h in self.H:
            fr = h.get("frac")
            if isinstance(fr, list):
                pct = " = " + " / ".join(f"{100 * x:.1f}%" if _fin(x) else "nan" for x in fr)
            else:
                pct = f" = {100 * fr:.1f}%" if _fin(fr) else ""
            self.p(S, f"  {h['rule']:4s} {'/'.join(map(str, h['group'])):16s} {h['verdict']:8s} ({h['matched']}/{h['used']}{pct})")
        return self.H, self.T

    def section_text(self, keys):
        return "\n".join(l for k in keys for l in self.sec[k])


def analyze_and_save(plan, run_dir, start_infos, stage, test_results=None, timing=None, save=True):
    """SPEC9 10.1: the analysis of whatever is on disk; its text is APPENDED to analysis_output.txt, and results_part9.json and
    part9_outputs.zip are rewritten. Called automatically after each Part and by the analysis cell."""
    an = Analysis9(plan, run_dir, start_infos)
    Hs, Ts = an.run()
    with open(os.path.join(run_dir, "analysis_output.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n\n######## analysis after: {stage} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ########\n")
        f.write("\n".join(an.lines) + "\n")
    for k in ("A", "G", "R"):
        with open(os.path.join(run_dir, f"analysis_section_{k}.txt"), "w", encoding="utf-8") as f:
            f.write(an.section_text([k]) + "\n")
    if save:
        save_outputs(an, start_infos, test_results, timing, stage)
    return an



# =============================================================================================================
# SPEC9 sec. 8 new unit tests V1-V11 (+ UW). Ground truth: spec author's values (pre9.py / pre9b.py, verified).
# =============================================================================================================
def _even_measure(prob, st, d, s, runner):
    """Both signs t = F0 +- s d: runner(t) -> learner result; measure_ext on each; returns (even, [res+, res-], [sc+, sc-], vecs)."""
    R, S, V = [], [], []
    for sg in (1, -1):
        res = runner(st.F0 + sg * s * d)
        sc, vL, vNP, thNP, thTH = measure_ext(prob, st, res["theta"])
        R.append(res)
        S.append(sc)
        V.append(dict(vL=vL, vNP=vNP, thNP=thNP, thTH=thTH, thL=res["theta"]))
    ev = even_stats(V[0]["vL"], V[0]["vNP"], V[1]["vL"], V[1]["vNP"])
    return ev, R, S, V


def _runner(prob, st, L, s):
    return lambda t: run_learner9(prob, st, t, s, L)


def _ok_runs(R, S):
    """Learner runs CONVERGED (run status = the A3.1 status) and the nearest-point solves converged."""
    return all(r["status"] == "CONVERGED" for r in R) and all(x["np_status"] == "CONVERGED" for x in S)


def _stat(R):
    s = "/".join(r["status"] for r in R)
    if any(r.get("aj_lit_flags", 0) for r in R):
        s += (f" (A3.1: jumps with hz > 1e-7 under the original rule {'/'.join(str(r.get('aj_lit_flags', 0)) for r in R)}, "
              f"max hz - B {max(r.get('aj_hz_excess_max') or 0 for r in R):.1e})")
    if any(r.get("status_supp", r["status"]) != r["status"] for r in R):
        s += " (supplementary hz on the step vector: " + "/".join(r["status_supp"] for r in R) + \
             f"; max hz {max(r.get('aj_hz_max', 0) for r in R):.1e}, max step-vector hz {max(r.get('aj_hz_step_max', 0) for r in R):.1e})"
    return s


def _small_mlp(act):
    """196-32-32-10 MLP, n = 8, trained start (DJ(0.5) fit of the true labels), seed 0 - independent of MODE."""
    key = "small_" + act
    if key in _TEST_CACHE:
        return _TEST_CACHE[key]
    X, labels, Y = get_data(0, 8)
    prob, th = layered_problem(X, [196, 32, 32, CFG["N_CLASSES"]], act, 0, name=f"{act}-small")
    th, fi = fit_dj(prob, th, Y)
    st = Start(prob, th)
    dirs, dinfo, S0 = make_directions(st, 0, 8)
    out = dict(prob=prob, st=st, Y=Y, dirs=dirs, dinfo=dinfo, S0=S0, fit=fi, th=th)
    _TEST_CACHE[key] = out
    return out


V1_CASES = [(L_AJ(-1), -1.0), (L_AJ(0.25), 0.25), (L_AJ(0.5), 0.5), (L_AJ(1), 1.0), (L_AJ(2), 2.0),
            (L_SLa(2, 1), 0.75), (L_SLa(8, 2), 0.6875), (L_SLa(4, -1), 0.125), (L_SLa(2, 0.5), 0.5),
            (L_TJ(0.5, 0.5), 0.5), (L_TJ(1.5, 0.5), 0.5), (L_TJ(0.5, 1), 0.6), (L_TJ(1.5, 1), 0.7143), (L_TJ(8, 1), 0.9),
            (L_TJ(3, 0.25), -1.0), (L_TM(0.5, 0.5), 0.5), (L_TM(1.5, 0.5), 0.5), (L_TM(0.5, 1), 0.6), (L_TM(1.5, 1), 0.7143),
            (L_TM(6, 1), 0.875), (L_TM(1.5, 0), -1.0)]


def t_V1():
    prob, th0 = toy_problem()
    st = Start(prob, th0)
    d = torch.ones(1, dtype=DT, device=th0.device)
    s = 0.02
    ok, parts = True, []
    for L, val in V1_CASES:
        ev, R, S, _ = _even_measure(prob, st, d, s, _runner(prob, st, L, s))
        good = abs(ev["rho"] - val) <= 0.01 * max(1.0, abs(val)) and _ok_runs(R, S)
        ok &= good
        extra = f", {_stat(R)}"
        if L["kind"] == "TM":
            extra += f", steps {R[0]['steps']}/{R[1]['steps']}, inner caps {R[0]['tm_inner_capped']}/{R[1]['tm_inner_capped']}"
        parts.append(f"{learner_name9(L)} rho={ev['rho']:+.4f} (want {val:+.4f}, cos {ev['cos']:+.4f}{extra}) {'ok' if good else 'FAIL'}")
    return ok, "u^2 - v^2, theta0 = (1, 1), s = 0.02, even part, tol 0.01*max(1,|value|): " + "; ".join(parts)


def t_V2():
    u = _u6_setup()
    prob, st, s = u["prob"], u["st"], u["s"]
    checks = [(L_AJ(0.5), "q1", 0.5, 0.01), (L_AJ(0.5), "rand", 0.5, 0.01), (L_AJ(2), "q1", 2.0, 0.03), (L_AJ(2), "rand", 2.0, 0.03),
              (L_SLa(8, 1), "q1", 0.5625, 0.01), (L_SLa(8, 1), "rand", 0.5625, 0.01), (L_TJ(1.5, 1), "q1", 0.7143, 0.01),
              (L_TJ(1.5, 1), "rand", 0.7143, 0.01), (L_TM(1.5, 1), "q1", 0.7143, 0.02), (L_TM(1.5, 0.5), "q1", 0.5, 0.01)]
    ok, parts = True, []
    for L, dn, val, tol in checks:
        ev, R, S, _ = _even_measure(prob, st, u[dn], s, _runner(prob, st, L, s))
        good = abs(ev["rho"] - val) <= tol and _ok_runs(R, S)
        ok &= good
        parts.append(f"{learner_name9(L)} {dn}: rho={ev['rho']:+.4f} (want {val:+.4f} +-{tol}, cos {ev['cos']:+.4f}, "
                     f"{_stat(R)}) {'ok' if good else 'FAIL'}")
    return ok, f"U6 network, s = {s:.4f}, even part: " + "; ".join(parts)


def t_V3():
    u = _u6_setup()
    prob, st, s = u["prob"], u["st"], u["s"]
    t = st.F0 + s * u["q1"]
    parts, ok = [], True
    jr = jump(prob, st.th0, t)
    a0 = alpha_jump(prob, st.th0, t, 0.0)
    dz = vnorm(a0["theta"] - jr["theta"])
    g0 = dz <= 1e-10 and a0["status"] == "CONVERGED"
    ok &= g0
    parts.append(f"AJ(0) vs JUMP: |dtheta| = {dz:.1e} [<=1e-10] ({a0['outer_it']} outer its) {'ok' if g0 else 'FAIL'}")
    for al in (0.5, 1.0, 2.0):
        r = alpha_jump(prob, st.th0, t, al)
        g = r["status"] == "CONVERGED" and r["res_inf"] <= 1e-11 and r["hz"] <= 1e-7
        ok &= g
        parts.append(f"AJ({al:g}): res {r['res_inf']:.1e} [<=1e-11], hz {r['hz']:.1e} [<=1e-7], outer its {r['outer_it']}, "
                     f"dampings {r['dampings']} {'ok' if g else 'FAIL'}")
        if al == 1.0:
            np0 = nearest_point(prob, st.th0, prob.F(r["theta"]), st.th0)
            rel = vnorm(r["theta"] - np0["theta"]) / vnorm(np0["theta"] - st.th0)
            g1 = rel <= 1e-6 and np0["status"] == "CONVERGED"
            ok &= g1
            parts.append(f"AJ(1) vs nearest_point started from theta0: rel {rel:.1e} [<=1e-6] (NP {np0['status']}, {np0['it']} its) "
                         f"{'ok' if g1 else 'FAIL'}")
    # supplementary engineering checks: jump_a == jump and measure_ext == measure (bitwise)
    ja = jump_a(prob, st.th0, t)
    same_j = torch.equal(ja["theta"], jr["theta"]) and ja["status"] == jr["status"]
    thL = run_SL(prob, st, t, 8)["theta"]
    m1 = measure(prob, st, thL)
    m2 = measure_ext(prob, st, thL)
    same_m = (all((m1[0][k] == m2[0][k]) or (isinstance(m1[0][k], float) and math.isnan(m1[0][k]) and math.isnan(m2[0][k]))
                  for k in m1[0]) and torch.equal(m1[1], m2[1]) and torch.equal(m1[2], m2[2]))
    ok &= same_j and same_m
    parts.append(f"supplementary: jump_a == jump bitwise: {same_j}; measure_ext == measure (all scalars, v_L, v_NP) bitwise: {same_m}")
    return ok, "U6 network, direction q1, + sign: " + "; ".join(parts)


def t_V4():
    cb = linear_combo()
    prob, th, Y, labels = combo_problem(cb)
    st = Start(prob, th)
    dirs, _, S0 = make_directions(st, 0, CFG["N_TRAIN"])
    t = st.F0 + 1.0 * dirs["rand"]
    r0 = st.F0 - t
    c = 0.7
    parts, worst_tm, worst_tj = [], 0.0, 0.0
    keep = (CFG["TM_CAP"], CFG["TJ_CAP"])
    try:
        CFG["TM_CAP"], CFG["TJ_CAP"] = 1, 1                    # exactly ONE step
        for al in (0.0, 0.5, 1.0):
            res = run_TM(prob, st, t, 1.0, c, al)
            assert res["steps"] == 1
            r1 = prob.F(res["theta"]) - t
            for i in (1, 2, 5):
                q = st.Q[:, i - 1]
                ci = c * float(st.lam[i - 1]) / st.lam1
                fac = (1 - ci * (1 - al)) / (1 + ci * al)
                want = fac * float(q @ r0)
                rel = abs(float(q @ r1) - want) / abs(want)
                worst_tm = max(worst_tm, rel)
            parts.append(f"TM(0.7,{al:g}) one step: max rel err over i in (1,2,5) so far {worst_tm:.1e}")
        for al in (0.5, 1.0):
            res = run_TJ(prob, st, t, c, al)
            assert res["steps"] == 1
            q = (1 - c * (1 - al)) / (1 + c * al)
            r1 = prob.F(res["theta"]) - t
            rel = vnorm(r1 - q * r0) / vnorm(q * r0)
            worst_tj = max(worst_tj, rel)
            parts.append(f"TJ(0.7,{al:g}) one step: |r1 - q r0|/|q r0| = {rel:.1e} (q = {q:.4f})")
    finally:
        CFG["TM_CAP"], CFG["TJ_CAP"] = keep
    ok = worst_tm <= 1e-9 and worst_tj <= 1e-8
    return ok, "linear control: " + "; ".join(parts) + f" [TM <= 1e-9, TJ <= 1e-8]"


def _log_check(prob, st, t, L):
    log = []
    res = run_logged(prob, st, t, L, log, keep_points=True)
    tot = torch.zeros_like(st.th0)
    for it in log:
        J = prob.J(it["the"])
        tot = tot + J.T @ csolve(chol(gram(J)), it["b"])
    a, b = st.V0(tot), st.V0(res["theta"] - st.th0)
    return vnorm(a - b) / vnorm(b), res, len(log)


def t_V5():
    u = _u6_setup()
    sm = _small_mlp("relu")
    V5_FP = 2.0                                         # A5: rounding-aware bound, local constant
    worst, parts, ok = 0.0, [], True                    # worst = max (rel err)/(bound)

    def bound(st_, res_, nlog_):
        vL = st_.V0(res_["theta"] - st_.th0)
        return 1e-8 + V5_FP * nlog_ * FP_EPS * vnorm(res_["theta"]) / vnorm(vL)
    for nm, prob, st, t in (("U6 q1", u["prob"], u["st"], u["st"].F0 + u["s"] * u["q1"]),
                            ("ReLU 196-32-32-10 eigA", sm["prob"], sm["st"], sm["st"].F0 + 1e-2 * sm["S0"] * sm["dirs"]["eigA"])):
        relu = nm.startswith("ReLU")
        for L in (L_SL(8), L_DJ(0.5), L_AJ(0.5), L_SLa(8, 1)):
            rel, res, nlog = _log_check(prob, st, t, L)
            if relu and L["kind"] in ("AJ", "SLa") and res["status"] == "AJ-FAILED":     # A4: allowed on ReLU
                parts.append(f"{nm} {learner_name9(L)}: AJ-FAILED (allowed on ReLU: no self-consistent alpha-point at a kink, A4) "
                             f"aj_outer_max {res.get('aj_outer_max')}, aj_dampings {res.get('aj_dampings')}, aj_why '{res.get('aj_why')}'")
                continue
            bd = bound(st, res, nlog)
            worst = max(worst, rel / bd)
            parts.append(f"{nm} {learner_name9(L)}: rel err {rel:.1e} [<= {bd:.1e}] ({nlog} steps, {res['status']})")
            ok &= res["status"] == "CONVERGED" and rel <= bd
        a = run_DJ(prob, st, t, 0.5)["theta"]
        b = run_DJ_logged(prob, st, t, 0.5, [])["theta"]
        c1 = run_SL(prob, st, t, 8)["theta"]
        c2 = run_SL_logged(prob, st, t, 8, [])["theta"]
        same = torch.equal(a, b)
        ok &= same
        parts.append(f"{nm}: run_DJ_logged == run_DJ weights bitwise: {same}; supplementary run_SL_logged == run_SL: {torch.equal(c1, c2)}")
    # A4 (ii): AJ(0.5), SLa(8,1) on the ReLU MLP at the no-switch size found exactly as in V7
    prob, st, S0, d = sm["prob"], sm["st"], sm["S0"], sm["dirs"]["eigA"]
    frac, found = 1e-3, False
    for k in range(5):
        ref = RefPath(prob, st, d, frac * S0, CFG["G_REF"])
        if ref.complete() and sum(ref.cell_change) == 0:
            found = True
            break
        if k == 4:
            break
        frac /= 10.0
    ref = None
    if not found:
        ok = False
        parts.append("ReLU (ii): reference path still has switches on [0, 2] after 4 divisions FAIL")
    else:
        t = st.F0 + frac * S0 * d
        for L in (L_AJ(0.5), L_SLa(8, 1)):
            rel, res, nlog = _log_check(prob, st, t, L)
            bd = bound(st, res, nlog)
            worst = max(worst, rel / bd)
            ok &= res["status"] == "CONVERGED" and rel <= bd
            parts.append(f"ReLU (ii) no-switch size s = {frac:g} S0 {learner_name9(L)}: rel err {rel:.1e} [<= {bd:.1e}] ({nlog} steps, {res['status']})")
    return ok, ("|V0 sum_k J(th_e)^T K(th_e)^-1 b_k - V0(th_L - th0)| / |V0(th_L - th0)| [<= 1e-8 + 2 m eps |th_L| / |v_L| (A5)]: "
                + "; ".join(parts) + f"; largest (rel err)/(bound) = {worst:.3f}")


def t_V6():
    u = _u6_setup()
    prob, st, s = u["prob"], u["st"], u["s"]
    d = u["q1"]
    ref = RefPath(prob, st, d, s, CFG["G_REF"])
    if not ref.complete():
        return False, f"reference path failed at j={ref.fail_j}"
    vPP, om, omp = R_profile(ref)
    dev = max(abs(om[k] - k / 16.0) for k in range(17))
    ok = dev <= 0.02
    parts = [f"G_REF = {CFG['G_REF']}; max_tau in [0,1] |omega - tau| = {dev:.2e} [<=0.02]"]
    for L in (L_SL(8), L_DJ(0.5)):
        sc, _ = R_learner(prob, st, ref, vPP, st.F0 + s * d, L)
        g = sc["cosF"] >= 0.9999 and abs(sc["nrF"] - 1) <= 0.01
        ok &= g
        parts.append(f"{sc['learner']}: cosF {sc['cosF']:.6f} [>=0.9999], nrF {sc['nrF']:.5f} [|.-1|<=0.01], rho_PP {sc['rho_PP']:+.4f}, "
                     f"rho_field {sc['rho_field']:+.4f}, rho_omega {sc['rho_omega']:+.4f} {'ok' if g else 'FAIL'}")
    npr = nearest_point(prob, st.th0, st.F0 + s * d, st.th0)
    vNP = st.V0(npr["theta"] - st.th0)
    cs = cosine(vPP, vNP)
    ok &= cs >= 0.999
    parts.append(f"cos(v_PP, v_NP) = {cs:.6f} [>=0.999], |v_PP|/|v_NP| = {vnorm(vPP) / vnorm(vNP):.5f}")
    return ok, "U6 network, q1, + sign: " + "; ".join(parts)


def t_V7():
    sm = _small_mlp("relu")
    prob, st, S0 = sm["prob"], sm["st"], sm["S0"]
    d = sm["dirs"]["eigA"]
    frac, ref = 1e-3, None
    for k in range(5):
        ref = RefPath(prob, st, d, frac * S0, CFG["G_REF"])
        if ref.complete() and sum(ref.cell_change) == 0:
            break
        if k == 4:
            return False, f"reference path still has switches on [0, 2] at s = {frac:g} S0 after 4 divisions"
        frac /= 10.0
    vPP, om, omp = R_profile(ref)
    ok, parts = True, [f"size used s = {frac:g} S0 (no switch on [0, 2])"]
    for L, val in ((L_SL(8), 0.4375), (L_DJ(1.5), -1.0)):
        sc, _ = R_learner(prob, st, ref, vPP, st.F0 + frac * S0 * d, L)
        g = abs(sc["rho_PP"] - val) <= 0.01 * max(1.0, abs(val)) and sc["learner_status"] == "CONVERGED"
        ok &= g
        parts.append(f"{sc['learner']}: rho_PP {sc['rho_PP']:+.4f} (want {val:+.4f}), cosF {sc['cosF']:.5f} {'ok' if g else 'FAIL'}")
    return ok, "small ReLU MLP, eigA, + sign: " + "; ".join(parts)


def t_V8():
    sm = _small_mlp("relu")
    prob, st, S0 = sm["prob"], sm["st"], sm["S0"]
    s = 1e-2 * S0
    ok, parts, sw_signs = True, [], []
    for sg in (1, -1):
        d = sg * sm["dirs"]["eigA"]
        ref = RefPath(prob, st, d, s, CFG["G_REF"])
        if not ref.complete():
            return False, f"reference path failed (sign {sg:+d})"
        vPP, om, omp = R_profile(ref)
        has = ref.switch_on(1.0)
        for L in (L_SL(8), L_SL(32), L_DJ(0.5), L_DJ(1.0)):
            sc, _ = R_learner(prob, st, ref, vPP, st.F0 + s * d, L)
            g = sc["cosF"] >= 0.98 and 0.95 <= sc["nrF"] <= 1.05
            ok &= g
            parts.append(f"sign {sg:+d} {sc['learner']}: cosF {sc['cosF']:.5f}, nrF {sc['nrF']:.4f}, rho_PP {sc['rho_PP']:+.4f} {'ok' if g else 'FAIL'}")
            if has and sc["learner"] == "SL(8)":
                miss = abs(sc["rho_PP"] - 0.4375) > 0.05
                ok &= miss
                sw_signs.append(sg)
                parts.append(f"sign {sg:+d} has a switch on [0,1] (cells {sum(ref.cell_change[:CFG['G_REF'] // 2])}): "
                             f"|rho_PP(SL(8)) - 0.4375| = {abs(sc['rho_PP'] - 0.4375):.4f} [> 0.05] {'ok' if miss else 'FAIL'}")
    if not sw_signs:
        ok = False
        parts.append("neither sign has a switch on [0, 1] (the second condition cannot be evaluated) FAIL")
    return ok, f"small ReLU MLP, eigA, s = 1e-2 S0, G_REF = {CFG['G_REF']}: " + "; ".join(parts)


def t_V9():
    sm = _small_mlp("tanh")
    prob, st, S0 = sm["prob"], sm["st"], sm["S0"]
    Xh = torch.randn(200, 196, generator=gen(9100), dtype=DT).to(DEVICE)
    d, s = sm["dirs"]["eigA"], 0.01 * S0
    ok, parts = True, []
    for L, val in ((L_SL(8), 0.4375), (L_DJ(1.5), -1.0), (L_AJ(2), 2.0)):
        ev, R, S, V = _even_measure(prob, st, d, s, _runner(prob, st, L, s))
        G = []
        for v in V:
            FT = test_outputs(prob, Xh, v["thTH"])
            G.append((test_outputs(prob, Xh, v["thL"]) - FT, test_outputs(prob, Xh, v["thNP"]) - FT))
        eg = even_stats(G[0][0], G[0][1], G[1][0], G[1][1])
        cs = eg["cos"] * math.copysign(1.0, val)            # sign-aware, as in matches() (A3.3)
        g = abs(eg["rho"] - val) <= 0.01 * max(1.0, abs(val)) and cs >= 0.999 and _ok_runs(R, S)
        ok &= g
        lit = "literal cos_test >= 0.999: " + ("holds" if eg["cos"] >= 0.999 else "cannot hold (f < 0)")
        parts.append(f"{learner_name9(L)}: gamma {eg['rho']:+.4f} (want {val:+.4f}), cos_test {eg['cos']:+.6f}, "
                     f"cos_test*sign(f) {cs:.6f} [>=0.999] ({lit}), rho {ev['rho']:+.4f} {'ok' if g else 'FAIL'}")
    return ok, "small tanh MLP (n=8, trained), eigA, s = 0.01 S0, even part, 200 Gaussian held-out inputs: " + "; ".join(parts)


def t_V10():
    parts, ok = [], True
    for seed, n in sorted({(0, CFG["N_TRAIN"]), (1, CFG["N_TRAIN"]), (0, 8)}):
        x, m, sd = train_scalars(seed, n)
        X = get_data(seed, n)[0]
        td = get_test_data(seed, n)
        same_x = torch.equal(((x - m) / sd).to(X.device), X)
        same_s = (float(td["mean"]).hex() == float(m).hex() and float(td["std"]).hex() == float(sd).hex()
                  and torch.equal(td["mean"], m) and torch.equal(td["std"], sd))
        g = same_x and same_s
        ok &= g
        parts.append(f"seed {seed}, n {n}: training inputs reproduced bitwise from the stored scalars: {same_x}; scalars applied to the "
                     f"held-out inputs bit-identical (mean {float(m).hex()}, std {float(sd).hex()}): {same_s} ({td['n_test']} held-out inputs)")
    return ok, "; ".join(parts)


def _cmp_rec(a, b, path=""):
    """Max abs difference of numbers (and list of non-numeric mismatches) between two records."""
    skip = {"t_learner", "t_measure", "t_total", "t_ref", "t_learner_pred", "wall_s", "finished", "worker", "device"}
    worst, bad = 0.0, []
    if isinstance(a, dict):
        for k in a:
            if k in skip:
                continue
            w, bb = _cmp_rec(a[k], b.get(k) if isinstance(b, dict) else None, path + "." + str(k))
            worst, bad = max(worst, w), bad + bb
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return worst, [path]
        for i, (x, y) in enumerate(zip(a, b)):
            w, bb = _cmp_rec(x, y, f"{path}[{i}]")
            worst, bad = max(worst, w), bad + bb
    elif isinstance(a, float) and not isinstance(a, bool):
        if not isinstance(b, (int, float)):
            return worst, [path]
        if math.isnan(a) and math.isnan(b):
            return 0.0, []
        worst = abs(a - b) if math.isfinite(a) and math.isfinite(b) else (0.0 if a == b else float("inf"))
    elif a != b:
        bad = [path]
    return worst, bad


def t_V11():
    parts, ok = [], True
    sm = _small_mlp("tanh")
    cb = dict(act="tanh", start="trained", seed=0, n=8)
    ctx = Ctx9(cb, prob=sm["prob"], th0=sm["th"], Y=sm["Y"])
    case = _case9(ctx.cname, "A", "A", L_AJ(0.5), "eigA", "s1", CFG["S1_FRAC"], 1)
    A = [run_case9(ctx, case) for _ in range(2)]
    w, bad = _cmp_rec(A[0][0], A[1][0])
    vd = max(vnorm(A[0][1][k] - A[1][1][k]) for k in A[0][1])
    g = w <= 1e-10 and not bad and vd <= 1e-10
    ok &= g
    parts.append(f"Part A case {case['name']} twice: max |diff| over scalars {w:.1e}, vectors {vd:.1e}, mismatches {bad or 'none'} "
                 f"(rho {A[0][0]['rho']:.15f} / {A[1][0]['rho']:.15f}) {'ok' if g else 'FAIL'}")
    sr = _small_mlp("relu")
    cbr = dict(act="relu", start="trained", seed=0, n=8)
    ctxr = Ctx9(cbr, prob=sr["prob"], th0=sr["th"], Y=sr["Y"])
    caser = _case9(ctxr.cname, "R", "R", None, "eigA", "r0.01", 1e-2, -1)
    Rr = [run_case9(ctxr, caser) for _ in range(2)]
    w, bad = _cmp_rec(Rr[0][0], Rr[1][0])
    vd = max(vnorm(Rr[0][1][k] - Rr[1][1][k]) for k in Rr[0][1])
    g = w <= 1e-10 and not bad and vd <= 1e-10
    ok &= g
    parts.append(f"Part R case {caser['name']} twice (all {len(Rr[0][0]['learners'])} learners): max |diff| over scalars {w:.1e}, "
                 f"vectors {vd:.1e}, mismatches {bad or 'none'} {'ok' if g else 'FAIL'}")
    return ok, "; ".join(parts)


def t_V12():
    """A3.6: rounding-aware AJ consistency (A3.1) incl. a negative control, and the TM inner stop with floor (A3.2)."""
    u = _u6_setup()
    prob, st, s = u["prob"], u["st"], u["s"]
    t = st.F0 + s * u["q1"]
    parts, ok = [], True
    r = run_TJ(prob, st, t, 0.5, 0.5)
    ga = r["status"] == "CONVERGED" and r["aj_lit_flags"] >= 1 and r["aj_hz_excess_max"] <= 1e-9
    ok &= ga
    parts.append(f"(a) TJ(0.5,0.5): status {r['status']} [CONVERGED], aj_lit_flags {r['aj_lit_flags']} [>=1] of {r['aj_calls']} jumps, "
                 f"aj_hz_excess_max {r['aj_hz_excess_max']:.2e} [<=1e-9], max hz {r['aj_hz_max']:.2e} {'ok' if ga else 'FAIL'}")
    aj = alpha_jump(prob, st.th0, t, 0.5)
    th = aj["theta"]
    the = st.th0 + 0.5 * (th - st.th0)
    Je = prob.J(the)
    w = torch.randn(prob.P, generator=gen(9200), dtype=DT).to(th.device)
    vw = w - Je.T @ csolve(chol(gram(Je)), Je @ w)
    Je = None
    uu = vw / vnorm(vw)
    th_bad = th + 1e-5 * vnorm(th - st.th0) * uu
    cb = aj_consistency(prob, st.th0, th_bad, 0.5)
    cg = aj_consistency(prob, st.th0, th, 0.5)
    gb = aj["status"] == "CONVERGED" and cb["inconsistent"] and 0.9e-5 <= cb["hz"] <= 1.1e-5 and not cg["inconsistent"]
    ok &= gb
    parts.append(f"(b) negative control: AJ(0.5) {aj['status']}; theta_bad: inconsistent {cb['inconsistent']} [True], hz {cb['hz']:.4e} "
                 f"[0.9e-5, 1.1e-5], B {cb['hz_bound']:.1e}; theta: inconsistent {cg['inconsistent']} [False], hz {cg['hz']:.1e}, "
                 f"B {cg['hz_bound']:.1e} {'ok' if gb else 'FAIL'}")
    tp, t0 = toy_problem()
    stt = Start(tp, t0)
    d = torch.ones(1, dtype=DT, device=t0.device)
    rm = run_TM(tp, stt, stt.F0 + 0.02 * d, 0.02, 6.0, 1.0)
    gc_ = rm["status"] == "CONVERGED" and rm["tm_inner_capped"] == 0 and rm["tm_inner_total"] <= 3 * rm["steps"]
    ok &= gc_
    parts.append(f"(c) TM(6,1) on u^2 - v^2, + sign: status {rm['status']} [CONVERGED], tm_inner_capped {rm['tm_inner_capped']} [0], "
                 f"tm_inner_total {rm['tm_inner_total']} [<= 3 x {rm['steps']} steps], floor accepts {rm['tm_inner_floor_accepts']}, "
                 f"max |g|/(eps|theta|) {rm['tm_floor_g_max']:.2f} {'ok' if gc_ else 'FAIL'}")
    return ok, "; ".join(parts)


def t_UW(run_dir):
    """Worker identity: one Part A case in the kernel and in a worker subprocess; every stored number must be identical."""
    slots, why = parallel_slots()
    ng = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ng >= 1:
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    elif slots:
        dev = "cpu"
    else:
        return None, "skipped (no GPU and no CPU workers requested)"
    a0 = CFG["A_COMBOS"][0][0]
    cb = dict(act=a0, start="init", seed=0)
    case = [c for c in plan_A(cb) if c["learner"] == "AJ(0.5)" and c["dir"] == "label" and c["size"] == "s1" and c["sign"] == 1][0]
    base = os.path.join(run_dir, "_worker_identity_test")
    if os.path.exists(base):
        shutil.rmtree(base)
    dK, dW = os.path.join(base, "kernel"), os.path.join(base, "worker")
    build_or_load_start(cb, dK, verbose=False)
    run_subdir(dW, "starts")
    for f in glob.glob(os.path.join(dK, "starts", "*")):
        shutil.copy(f, os.path.join(dW, "starts"))
    sig = config_signature()
    run_combo_cases9(cb, [case], dK, sig, worker="kernel-test")
    write_run_files(dW, [dict(combo=cb, cases=[case])], "UW")
    P = _spawn(dW, 0, 1, dev, torch.get_num_threads(), "UW")
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
                f"{rw['rho']!r}, gamma-side |G_L| kernel {rk['nGL']!r} worker {rw['nGL']!r}; differing scalars: {diffs or 'none'}; "
                f"differing vectors: {vdiff or 'none'}")


TESTS_U = [("U1 Jacobian vs finite differences (each activation)", t_U1), ("U2 projector", t_U2), ("U3 JUMP", t_U3),
           ("U4 nearest point", t_U4), ("U5 exactly solvable model u^2 - v^2", t_U5),
           ("U6 small tanh network", t_U6), ("U7 linear control (20 parameters)", t_U7),
           ("U8 step-factor mapping", t_U8), ("U9 path bookkeeping identity", t_U9),
           ("U10 LAW on a straight path", t_U10), ("U11 LAW index order", t_U11), ("U12 determinism", t_U12),
           ("U13 SVD truncation in LAW", t_U13), ("U14 direction-converged stop rule (A2.2)", t_U14)]
TESTS_V = [("V1 solvable model: AJ, SLa, TJ, TM", t_V1), ("V2 U6 network: AJ, SLa, TJ, TM", t_V2),
           ("V3 AJ consistency", t_V3), ("V4 residual recursions (linear control)", t_V4),
           ("V5 step-log exactness (Part R logging)", t_V5), ("V6 reference-path prediction, smooth", t_V6),
           ("V7 ReLU inside a linear region", t_V7), ("V8 ReLU with switches", t_V8), ("V9 held-out family", t_V9),
           ("V10 test-set standardisation", t_V10), ("V11 determinism (Part A case, Part R case)", t_V11),
           ("V12 A3: rounding-aware AJ check (with negative control) and TM inner floor", t_V12)]


def run_tests(run_dir):
    """Runs U1-U14, V1-V11 and UW; writes part9_run/tests_output.txt and tests_results.json IMMEDIATELY."""
    results, lines = [], []
    t_all = time.time()
    hdr = (f"TESTS (SPEC9 sec. 8) MODE={CFG['MODE']} data_source={DATA_SOURCE} device={DEVICE} "
           f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(hdr)
    lines.append(hdr)
    for name, fn in TESTS_U + TESTS_V + [("UW worker identity (kernel vs worker subprocess)", lambda: t_UW(run_dir))]:
        t1 = time.time()
        try:
            ok, msg = fn()
        except Exception:
            ok, msg = False, "EXCEPTION:\n" + traceback.format_exc()
        tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        txt = f"{tag}  {name}  ({time.time() - t1:.1f}s)\n      {msg}"
        say(txt)
        lines.append(txt)
        results.append(dict(test=name, result=tag, detail=msg, seconds=time.time() - t1))
    tail = f"TESTS finished in {fmt_time(time.time() - t_all)}"
    say(tail)
    lines.append(tail)
    lines.append("TESTS SUMMARY: " + ", ".join(f"{r['test'].split()[0]}={r['result']}" for r in results))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "tests_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    write_json_atomic(os.path.join(run_dir, "tests_results.json"), results)
    return results



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


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def family(case):
    if case["kind"] == "R":
        return "R"
    if case["kind"] == "G4":
        return "G4"
    k = case["L"]["kind"]
    if case["kind"] == "lin":
        return "lin"
    if k in ("SL", "DJ"):
        return "A-anchor"
    return "A-" + k


def timing_estimate(plan, run_dir, n_gpus=1):
    """Measured cost of one case of each kind (run once, NOT written to done/) and an estimate per Part.
    The grid is never changed because of the estimate."""
    out = dict(cases={}, parts={})
    say("timing: one case of each kind (results are discarded; the grid runs them again)")
    samples = {}
    for part in PARTS:
        for it in plan.get(part, []):
            for c in it["cases"]:
                f = family(c)
                if f not in samples and c["sign"] == 1:
                    samples[f] = (it["combo"], c)
    ctxs = {}
    tJ = None
    for f, (cb, c) in samples.items():
        nm = cname9(cb)
        if nm not in ctxs:
            ctxs.clear()
            free_memory()
            ctxs[nm] = Ctx9(cb, run_dir)
        ctx = ctxs[nm]
        if tJ is None:
            _sync()
            t1 = time.time()
            gram(ctx.prob.J(ctx.st.th0))
            _sync()
            tJ = time.time() - t1
            out["t_J"] = tJ
            say(f"  one Jacobian + K on {DEVICE} ({nm}, N={ctx.prob.N}, P={ctx.prob.P}): {tJ * 1e3:.1f} ms")
        c0 = ctx.prob.counts["J"]
        _sync()
        t1 = time.time()
        try:
            rec, _ = run_case9(ctx, c)
            st_ = rec.get("learner_status", rec.get("ref_status"))
        except Exception as e:
            st_ = f"ERROR {e!r}"
        _sync()
        dt = time.time() - t1
        nJ = ctx.prob.counts["J"] - c0
        out["cases"][f] = dict(case=c["name"], seconds=dt, n_J=nJ, status=st_)
        say(f"  {f:10s} {c['name']}: {fmt_time(dt)} ({nJ} Jacobians, status {st_})")
    ctxs.clear()
    free_memory()
    ng = max(1, n_gpus)
    tot1 = totn = 0.0
    for part in PARTS:
        items = plan.get(part, [])
        per_combo = []
        for it in items:
            s = 0.0
            for c in it["cases"]:
                x = out["cases"].get(family(c))
                s += x["seconds"] if x else 0.0
            per_combo.append(s)
        one = sum(per_combo)
        if part == "lin" or ng == 1:
            many = one
        else:
            loads = [sum(per_combo[i] for i in range(len(per_combo)) if i % ng == k) for k in range(ng)]
            many = max(loads) if loads else 0.0
        out["parts"][part] = dict(one_device=one, n_devices=many, cases=sum(len(it["cases"]) for it in items))
        tot1 += one
        totn += many
        say(f"  estimate Part {part:9s}: {out['parts'][part]['cases']:5d} cases, {fmt_time(one)} on one device, "
            f"{fmt_time(many)} with {ng} worker(s) (static split by combination)")
    say(f"  estimate total: {fmt_time(tot1)} on one device; {fmt_time(totn)} with {ng} worker(s). Caveat: one sample per kind; "
        f"TM cases cost about (steps) x 1-3 Jacobians.")
    out.update(total_one=tot1, total_n=totn, n_devices=ng)
    return out


def save_outputs(an, start_infos, test_results, timing, stage):
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               device=str(DEVICE), gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
               if torch.cuda.is_available() else [], platform=platform.platform(), threads=torch.get_num_threads())
    res = dict(spec="SPEC9", mode=CFG["MODE"], data_source=DATA_SOURCE, signature=config_signature(), stage=stage,
               written=time.strftime("%Y-%m-%d %H:%M:%S"), config={k: v for k, v in CFG.items()}, environment=env,
               tests=test_results, starts=start_infos, timing=timing, H_table=an.H, report_tables=an.T,
               failed=[dict(name=f["name"], error=f["error"]) for f in an.failed], missing=an.missing,
               cases=an.recs, analysis_text="\n".join(an.lines))
    write_json_atomic(CFG["RESULTS_FILE"], res)
    rd = CFG["RUN_DIR"]
    tmpz = CFG["ZIP_FILE"] + f".tmp{os.getpid()}"
    with zipfile.ZipFile(tmpz, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CFG["RESULTS_FILE"])
        for p in ("analysis_output.txt", "tests_output.txt", "tests_results.json", "progress.log",
                  "analysis_section_A.txt", "analysis_section_G.txt", "analysis_section_R.txt"):
            q = os.path.join(rd, p)
            if os.path.exists(q):
                z.write(q)
        for sub in ("done", "failed", "starts", "logs"):
            for p in sorted(glob.glob(os.path.join(rd, sub, "*"))):
                if sub == "starts" and p.endswith(".pt"):
                    continue
                z.write(p)
    os.replace(tmpz, CFG["ZIP_FILE"])
    say(f"[{stage}] wrote {os.path.abspath(CFG['RESULTS_FILE'])} ({os.path.getsize(CFG['RESULTS_FILE']) / 1e6:.2f} MB) and "
        f"{os.path.abspath(CFG['ZIP_FILE'])} ({os.path.getsize(CFG['ZIP_FILE']) / 1e6:.2f} MB; vecs/ not included)")
    return res

