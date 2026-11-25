"""Task-cycle rotation: model, tasks, learners, sequence runner, measurements.

Implements SPEC.md sections 2-4. No experiment logic here.
All arithmetic float64, CPU, deterministic.
"""
import math
import time

import numpy as np
import torch
from torch.func import jacrev

torch.set_default_dtype(torch.float64)

D_IN = 8          # input dimension d
N_TASK = 16       # inputs per task n
N_PROBE = 256     # probe inputs
TEACHER_H = 16    # teacher hidden width
LIN_FEATURES = 400

GN_TOL = 1e-12
GN_MAX_ITER = 100
GN_MAX_HALVINGS = 30
GD_TOL = 1e-8
GD_CAP = 400_000

SEQUENCES = {
    "retrace": ("B", "A"),
    "loop+": ("B", "C", "A"),
    "loop-": ("C", "B", "A"),
    "star+": ("B", "A", "C", "A"),
    "star-": ("C", "A", "B", "A"),
    "ff": ("B", "C", "A", "B", "C", "A"),
    "fr": ("B", "C", "A", "C", "B", "A"),
}


class NotConverged(RuntimeError):
    """Raised by the GN learner when it fails to converge (condition INVALID)."""


class Timeout(RuntimeError):
    """Raised by the GD learner when a wall-clock deadline is passed."""


# ---------------------------------------------------------------------------
# 2.1 Student MLP (NTK parametrisation), parameters as one flat vector theta
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, h, d=D_IN):
        self.h, self.d = h, d
        shapes = [("W1", (h, d)), ("b1", (h,)), ("W2", (h, h)), ("b2", (h,)),
                  ("w3", (h,)), ("b3", ())]
        self.slices = {}
        off = 0
        for name, shp in shapes:
            size = int(math.prod(shp)) if len(shp) else 1
            self.slices[name] = (off, off + size, shp)
            off += size
        self.P = off
        assert self.P == h * d + h + h * h + h + h + 1
        self.sd, self.sh = math.sqrt(d), math.sqrt(h)

    def unflat(self, theta):
        out = {}
        for name, (a, b, shp) in self.slices.items():
            out[name] = theta[a:b].reshape(shp) if len(shp) else theta[a]
        return out

    def init(self, seed):
        """theta_init for (seed, h): W1, W2, w3 ~ N(0,1) (drawn in that order), biases 0."""
        g = torch.Generator().manual_seed(1_000_003 * (seed + 1) + self.h)
        theta = torch.zeros(self.P)
        p = self.unflat(theta)
        p["W1"].copy_(torch.randn(self.h, self.d, generator=g))
        p["W2"].copy_(torch.randn(self.h, self.h, generator=g))
        p["w3"].copy_(torch.randn(self.h, generator=g))
        return theta

    def f(self, theta, X):
        p = self.unflat(theta)
        z1 = X @ p["W1"].T / self.sd + p["b1"]
        z2 = torch.tanh(z1) @ p["W2"].T / self.sh + p["b2"]
        return torch.tanh(z2) @ p["w3"] / self.sh + p["b3"]

    def jac(self, theta, X):
        """Exact Jacobian d f(X;theta)/d theta, shape (m, P), via torch.func.jacrev."""
        return jacrev(lambda t: self.f(t, X))(theta)

    def grad_loss(self, theta, X, y, out=None):
        """Manual backprop of L = (1/2n)||f - y||^2. Returns (grad, r)."""
        n = X.shape[0]
        p = self.unflat(theta)
        a1 = torch.tanh(X @ p["W1"].T / self.sd + p["b1"])
        a2 = torch.tanh(a1 @ p["W2"].T / self.sh + p["b2"])
        fv = a2 @ p["w3"] / self.sh + p["b3"]
        r = fv - y
        e = r / n
        g = torch.empty(self.P) if out is None else out
        q = self.unflat(g)
        q["w3"].copy_(a2.T @ e / self.sh)
        g[self.slices["b3"][0]] = e.sum()
        dz2 = (e[:, None] * p["w3"][None, :] / self.sh) * (1.0 - a2 * a2)
        q["W2"].copy_(dz2.T @ a1 / self.sh)
        q["b2"].copy_(dz2.sum(0))
        dz1 = (dz2 @ p["W2"] / self.sh) * (1.0 - a1 * a1)
        q["W1"].copy_(dz1.T @ X / self.sd)
        q["b1"].copy_(dz1.sum(0))
        return g, r

    def make_np_grad(self, X, y):
        """Same manual backprop as grad_loss, in numpy (float64) for low per-step overhead.
        Returns closure grad(theta_np, g_np) -> r_np, writing the gradient into g_np."""
        Xn, yn = X.numpy(), y.numpy()
        n = Xn.shape[0]
        sl = {k: (a, b, shp) for k, (a, b, shp) in self.slices.items()}
        sd, sh = self.sd, self.sh
        b3i = sl["b3"][0]

        def view(v, k):
            a, b, shp = sl[k]
            return v[a:b].reshape(shp)

        def grad(t, g):
            W1, b1, W2, b2, w3 = (view(t, k) for k in ("W1", "b1", "W2", "b2", "w3"))
            a1 = np.tanh(Xn @ W1.T / sd + b1)
            a2 = np.tanh(a1 @ W2.T / sh + b2)
            r = a2 @ w3 / sh + t[b3i] - yn
            e = r / n
            view(g, "w3")[:] = a2.T @ e / sh
            g[b3i] = e.sum()
            dz2 = (e[:, None] * w3[None, :] / sh) * (1.0 - a2 * a2)
            view(g, "W2")[:] = dz2.T @ a1 / sh
            view(g, "b2")[:] = dz2.sum(0)
            dz1 = (dz2 @ W2 / sh) * (1.0 - a1 * a1)
            view(g, "W1")[:] = dz1.T @ Xn / sd
            view(g, "b1")[:] = dz1.sum(0)
            return r

        return grad


# ---------------------------------------------------------------------------
# Linear-in-theta model for the tests (section 7)
# ---------------------------------------------------------------------------
class LinearModel:
    """f(x) = phi(x) . theta,  phi(x) = tanh(W x / sqrt(d) + b) / sqrt(400)."""

    def __init__(self, seed, n_feat=LIN_FEATURES, d=D_IN):
        g = torch.Generator().manual_seed(777_000 + seed)
        self.W = torch.randn(n_feat, d, generator=g)
        self.b = torch.randn(n_feat, generator=g)
        self.theta_true = torch.randn(n_feat, generator=g)
        self.theta_init0 = torch.randn(n_feat, generator=g)
        self.P = n_feat
        self.sd = math.sqrt(d)
        self.sf = math.sqrt(n_feat)

    def init(self, seed=None):
        return self.theta_init0.clone()

    def phi(self, X):
        return torch.tanh(X @ self.W.T / self.sd + self.b) / self.sf

    def f(self, theta, X):
        return self.phi(X) @ theta

    def jac(self, theta, X):
        return self.phi(X)

    def labels(self, X):
        return self.phi(X) @ self.theta_true

    def grad_loss(self, theta, X, y, out=None):
        Phi = self.phi(X)
        r = Phi @ theta - y
        return Phi.T @ r / X.shape[0], r


# ---------------------------------------------------------------------------
# 2.2 / 2.3 / 2.4 Teacher, tasks, probe
# ---------------------------------------------------------------------------
class TaskFamily:
    """Per-seed random objects. Draw order from Generator(seed):
    M(16x8), c(16), v(16), X_A(16x8), U_raw(16x8), V_raw(16x8), X_probe(256x8)."""

    def __init__(self, seed):
        g = torch.Generator().manual_seed(seed)
        self.M = torch.randn(TEACHER_H, D_IN, generator=g)
        self.c = torch.randn(TEACHER_H, generator=g)
        self.v = torch.randn(TEACHER_H, generator=g)
        self.X_A = torch.randn(N_TASK, D_IN, generator=g)
        U_raw = torch.randn(N_TASK, D_IN, generator=g)
        V_raw = torch.randn(N_TASK, D_IN, generator=g)
        self.X_probe = torch.randn(N_PROBE, D_IN, generator=g)
        nA = torch.linalg.norm(self.X_A)  # Frobenius
        self.U = U_raw * nA / torch.linalg.norm(U_raw)
        Vp = V_raw - ((V_raw * self.U).sum() / (self.U * self.U).sum()) * self.U
        self.Uperp = Vp * nA / torch.linalg.norm(Vp)

    def teacher(self, X):
        return torch.tanh(X @ self.M.T / math.sqrt(D_IN) + self.c) @ self.v / math.sqrt(TEACHER_H)

    def X_B(self, eps):
        return self.X_A + eps * self.U

    def X_C(self, eps, gamma):
        return self.X_A + eps * (math.cos(gamma) * self.U + math.sin(gamma) * self.Uperp)

    def X_C_col(self, eps, alpha):
        return self.X_A + alpha * eps * self.U

    def tasks(self, eps, gamma=None, alpha=None, label_fn=None):
        """Return {'A':(X,y), 'B':(X,y), 'C':(X,y)}. Exactly one of gamma/alpha."""
        label_fn = label_fn or self.teacher
        XA, XB = self.X_A, self.X_B(eps)
        if alpha is not None:
            XC = self.X_C_col(eps, alpha)
        else:
            XC = self.X_C(eps, gamma)
        return {k: (X, label_fn(X)) for k, X in (("A", XA), ("B", XB), ("C", XC))}


# ---------------------------------------------------------------------------
# 3.1 GN learner
# ---------------------------------------------------------------------------
def gn_step(model, theta, X, y):
    """One damped min-norm Gauss-Newton step. Returns (theta_new, r_new, s, ls_failed)."""
    r = model.f(theta, X) - y
    J = model.jac(theta, X)
    Delta = J.T @ torch.linalg.solve(J @ J.T, r)
    nr = torch.linalg.norm(r)
    s = 1.0
    ls_failed = False
    for k in range(GN_MAX_HALVINGS + 1):
        th_new = theta - s * Delta
        r_new = model.f(th_new, X) - y
        if torch.linalg.norm(r_new) < nr:
            break
        if k == GN_MAX_HALVINGS:
            ls_failed = True  # literal reading: take the step with s = 2^-30 anyway
            break
        s *= 0.5
    return th_new, r_new, s, ls_failed


def gn_learn(model, theta, task, tol=GN_TOL, max_iter=GN_MAX_ITER):
    X, y = task
    r = model.f(theta, X) - y
    it, n_halv_fail, min_s = 0, 0, 1.0
    while r.abs().max().item() > tol:
        if it >= max_iter:
            raise NotConverged(f"GN not converged after {max_iter} iterations, "
                               f"||r||_inf={r.abs().max().item():.3e}")
        theta, r, s, lsf = gn_step(model, theta, X, y)
        n_halv_fail += int(lsf)
        min_s = min(min_s, s)
        it += 1
    return theta, {"iters": it, "res_inf": r.abs().max().item(), "converged": True,
                   "ls_fail": n_halv_fail, "min_s": min_s}


# ---------------------------------------------------------------------------
# 3.2 GD learner
# ---------------------------------------------------------------------------
def gd_learn(model, theta, task, tol=GD_TOL, cap=GD_CAP, deadline=None):
    X, y = task
    n = X.shape[0]
    J0 = model.jac(theta, X)
    lam_max = torch.linalg.eigvalsh(J0 @ J0.T)[-1].item()
    lr = 1.0 * n / lam_max
    t0 = time.time()
    if hasattr(model, "make_np_grad"):
        grad = model.make_np_grad(X, y)
        th = theta.numpy().copy()
        gbuf = np.empty(model.P)
    else:
        th = theta.clone()
        gbuf = torch.empty(model.P)

        def grad(t, g):
            gg, r = model.grad_loss(t, X, y, out=g)
            return r.numpy()
    steps = 0
    while True:
        r = grad(th, gbuf)
        res = float(np.abs(r).max())
        done = res <= tol
        if done or steps >= cap:
            out = torch.from_numpy(th.copy()) if isinstance(th, np.ndarray) else th
            return out, {"steps": steps, "res_inf": res, "converged": bool(done), "lr": lr,
                         "lam_max": lam_max, "time": time.time() - t0}
        if deadline is not None and (steps & 1023) == 0 and time.time() > deadline:
            raise Timeout(f"deadline passed after {steps} GD steps (res_inf={res:.3e})")
        th -= lr * gbuf
        steps += 1


# ---------------------------------------------------------------------------
# 4.1 Linearised learner
# ---------------------------------------------------------------------------
def make_lin_learner(model, theta_A, tasks):
    frozen = {}
    for k, (X, y) in tasks.items():
        J0 = model.jac(theta_A, X)
        r0 = model.f(theta_A, X) - y
        frozen[k] = (J0, r0)

    def learn_lin(theta, task_name):
        J0, r0 = frozen[task_name]
        rhs = r0 + J0 @ (theta - theta_A)
        return theta - J0.T @ torch.linalg.solve(J0 @ J0.T, rhs), {"iters": 1, "converged": True}

    return learn_lin


# ---------------------------------------------------------------------------
# Sequence runner (with exact prefix reuse; deterministic so results identical)
# ---------------------------------------------------------------------------
def run_sequences(learn_fn, theta_A, seq_names, sequences=SEQUENCES):
    """learn_fn(theta, task_name) -> (theta, info). Returns ({name: theta_s}, {name: [infos]})."""
    cache = {(): (theta_A, [])}
    out, infos = {}, {}
    for name in seq_names:
        seq = tuple(sequences[name])
        k = len(seq)
        while seq[:k] not in cache:
            k -= 1
        theta, inf = cache[seq[:k]]
        inf = list(inf)
        for j in range(k, len(seq)):
            theta, i = learn_fn(theta, seq[j])
            inf.append(dict(i, task=seq[j]))
            cache[seq[:j + 1]] = (theta, list(inf))
        out[name], infos[name] = theta, inf
    return out, infos


# ---------------------------------------------------------------------------
# Measurements (section 4)
# ---------------------------------------------------------------------------
def _n(v):
    return torch.linalg.norm(v).item()


def _ratio(a, b):
    return a / b if b != 0 else float("nan")


def cosine(a, b):
    na, nb = _n(a), _n(b)
    return (torch.dot(a, b).item() / (na * nb)) if na > 0 and nb > 0 else float("nan")


def displacement_scalars(model, theta_A, thetas, X_probe):
    """All norm-type scalars of section 4 available from the sequences present,
    in weight space (suffix none) and function space (suffix _fn)."""
    f_A = model.f(theta_A, X_probe)
    dl = {k: v - theta_A for k, v in thetas.items()}
    ph = {k: model.f(v, X_probe) - f_A for k, v in thetas.items()}
    out = {}
    for space, D in (("", dl), ("_fn", ph)):
        if "loop+" in D and "loop-" in D:
            a = (D["loop+"] - D["loop-"]) / 2
            s = (D["loop+"] + D["loop-"]) / 2
            out["N_a" + space] = _n(a)
            out["N_s" + space] = _n(s)
            if "ff" in D and "fr" in D:
                out["E_add" + space] = _ratio(_n((D["ff"] - D["fr"]) - 2 * a), _n(2 * a))
                out["E_shrink" + space] = _ratio(_n(D["fr"] - 2 * s), _n(2 * s))
        if "star+" in D and "star-" in D:
            out["N_star" + space] = _n((D["star+"] - D["star-"]) / 2)
        if "retrace" in D:
            out["N_ret" + space] = _n(D["retrace"])
    return out, dl, ph


def a_loop_vec(theta_A, thetas):
    return ((thetas["loop+"] - theta_A) - (thetas["loop-"] - theta_A)) / 2


def tang(model, theta_A, X_A, a):
    J_A = model.jac(theta_A, X_A)
    specn = torch.linalg.matrix_norm(J_A, ord=2).item()
    return _ratio(_n(J_A @ a), specn * _n(a))


def lin_dev(model, theta_A, theta_lp, tasks):
    vals = []
    for k in ("A", "B", "C"):
        X = tasks[k][0]
        J0 = model.jac(theta_A, X)
        J1 = model.jac(theta_lp, X)
        vals.append(_n(J1 - J0) / _n(J0))
    return max(vals)


def make_learn(model, tasks, learner="gn", deadline=None):
    if learner == "gn":
        return lambda th, k: gn_learn(model, th, tasks[k])
    if learner == "gd":
        return lambda th, k: gd_learn(model, th, tasks[k], deadline=deadline)
    raise ValueError(learner)
