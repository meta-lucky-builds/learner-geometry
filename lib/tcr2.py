"""Task-cycle rotation, part 2 (SPEC2.md): polygon families, forward/reverse runner,
part-R linear-regression model, schedules and measures.

Reuses tcr.py unchanged (imported). No experiment logic here.
All arithmetic float64, CPU, deterministic.
"""
import math

import numpy as np
import torch

import tcr

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Exact analytic Jacobian for the SPEC.md MLP (same model, same learner; only the
# Jacobian routine is a faster exact method). Checked against tcr.MLP.jac (jacrev)
# in test_part2.py (extra test X1).
# ---------------------------------------------------------------------------
class FastMLP(tcr.MLP):
    def jac(self, theta, X):
        p = self.unflat(theta)
        m = X.shape[0]
        a1 = torch.tanh(X @ p["W1"].T / self.sd + p["b1"])
        a2 = torch.tanh(a1 @ p["W2"].T / self.sh + p["b2"])
        J = torch.empty(m, self.P)
        g2 = p["w3"][None, :] * (1.0 - a2 * a2) / self.sh            # df/dz2  (m,h)
        g1 = (g2 @ p["W2"] / self.sh) * (1.0 - a1 * a1)              # df/dz1  (m,h)
        sl = self.slices
        h, d = self.h, self.d
        J[:, sl["W1"][0]:sl["W1"][1]] = (g1[:, :, None] * X[:, None, :] / self.sd).reshape(m, h * d)
        J[:, sl["b1"][0]:sl["b1"][1]] = g1
        J[:, sl["W2"][0]:sl["W2"][1]] = (g2[:, :, None] * a1[:, None, :] / self.sh).reshape(m, h * h)
        J[:, sl["b2"][0]:sl["b2"][1]] = g2
        J[:, sl["w3"][0]:sl["w3"][1]] = a2 / self.sh
        J[:, sl["b3"][0]] = 1.0
        return J


# ---------------------------------------------------------------------------
# F.2 polygon families (vertex lists exclude the start point p_0 = A = (0,0))
# ---------------------------------------------------------------------------
def poly_tri():
    return [(1.0, 0.0), (0.0, 1.0)]


def poly_circ(N):
    assert N >= 3
    return [(math.cos(2 * math.pi * k / N) - 1.0, math.sin(2 * math.pi * k / N))
            for k in range(1, N)]


def poly_rect(c):
    return [(1.0, 0.0), (1.0, float(c)), (0.0, float(c))]


# (0,0) means "learn A"
POLY_FIG8 = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (-1.0, 0.0), (0.0, 1.0)]


def poly_fig8():
    return list(POLY_FIG8)


def get_poly(name):
    if name == "tri":
        return poly_tri()
    if name.startswith("circ"):
        return poly_circ(int(name[4:]))
    if name.startswith("rect"):
        return poly_rect(float(name[4:]))
    if name == "fig8":
        return poly_fig8()
    raise ValueError(name)


def shoelace(pts):
    """Signed area of the closed polygon (0,0), pts[0], ..., pts[-1], back to (0,0)."""
    full = [(0.0, 0.0)] + list(pts)
    N = len(full)
    return 0.5 * sum(full[k][0] * full[(k + 1) % N][1] - full[(k + 1) % N][0] * full[k][1]
                     for k in range(N))


def predicted_area(name):
    if name == "tri":
        return 0.5
    if name.startswith("circ"):
        N = int(name[4:])
        return N / 2 * math.sin(2 * math.pi / N)
    if name.startswith("rect"):
        return float(name[4:])
    if name == "fig8":
        return 0.0
    raise ValueError(name)


class PlaneTasks:
    """Task at plane point (x,y): X = X_A + eps (x U + y U_perp), labels label_fn(X).
    The point (0,0) is task A itself (X_A, label_fn(X_A))."""

    def __init__(self, fam, eps, label_fn=None):
        self.fam, self.eps = fam, eps
        self.label_fn = label_fn or fam.teacher
        self.A = (fam.X_A, self.label_fn(fam.X_A))
        self._cache = {}

    def __call__(self, p):
        p = (float(p[0]), float(p[1]))
        if p == (0.0, 0.0):
            return self.A
        if p not in self._cache:
            X = self.fam.X_A + self.eps * (p[0] * self.fam.U + p[1] * self.fam.Uperp)
            self._cache[p] = (X, self.label_fn(X))
        return self._cache[p]


# ---------------------------------------------------------------------------
# Forward / reverse runner
# ---------------------------------------------------------------------------
def run_cycles(learn, theta, vtasks, A, m_list):
    """Repeat (learn vtasks in order, then A) max(m_list) times starting from theta.
    learn(theta, task) -> (theta, info). Returns ({m: theta after m cycles}, [infos])."""
    out, infos = {}, []
    M = max(m_list)
    for c in range(1, M + 1):
        for t in list(vtasks) + [A]:
            theta, info = learn(theta, t)
            infos.append(info)
        if c in m_list:
            out[c] = theta.clone()
    return out, infos


def polygon_run(model, learn, theta_A, pts, ptasks, X_probe, m_list=(1,)):
    """Forward = learn pts in order then A; reverse = learn pts reversed then A (each repeated
    m times). Returns {m: dict(a, s, a_fn, s_fn, d_plus, d_minus)}, infos_plus, infos_minus,
    thetas_plus, thetas_minus."""
    fwd = [ptasks(p) for p in pts]
    rev = [ptasks(p) for p in reversed(pts)]
    th_p, inf_p = run_cycles(learn, theta_A, fwd, ptasks.A, m_list)
    th_m, inf_m = run_cycles(learn, theta_A, rev, ptasks.A, m_list)
    f_A = model.f(theta_A, X_probe)
    res = {}
    for m in m_list:
        dp, dm = th_p[m] - theta_A, th_m[m] - theta_A
        pp, pm = model.f(th_p[m], X_probe) - f_A, model.f(th_m[m], X_probe) - f_A
        res[m] = {"a": (dp - dm) / 2, "s": (dp + dm) / 2,
                  "a_fn": (pp - pm) / 2, "s_fn": (pp + pm) / 2,
                  "d_plus": dp, "d_minus": dm}
    return res, inf_p, inf_m, th_p, th_m


# ---------------------------------------------------------------------------
# Part R: plain linear regression with cyclic projections
# ---------------------------------------------------------------------------
class RModel:
    """theta* ~ N(0, I_d); X_t (n x d) ~ N(0,1); y_t = X_t theta*.
    Draw order from Generator(seed): theta*, then X_0, ..., X_{T-1}."""

    def __init__(self, seed, T, n, d):
        g = torch.Generator().manual_seed(seed)
        self.T, self.n, self.d = T, n, d
        self.theta_star = torch.randn(d, generator=g)
        self.X = torch.stack([torch.randn(n, d, generator=g) for _ in range(T)])  # (T,n,d)
        self.y = torch.einsum("tnd,d->tn", self.X, self.theta_star)
        self.pinv = torch.stack([torch.linalg.pinv(self.X[t]) for t in range(T)])  # (T,d,n)
        self.lam_max = torch.stack([torch.linalg.eigvalsh(self.X[t] @ self.X[t].T)[-1]
                                    for t in range(T)])

    def losses(self, theta):
        """(L_0..L_{T-1}),  L_t = (1/n) ||X_t (theta - theta*)||^2."""
        r = torch.einsum("tnd,d->tn", self.X, theta - self.theta_star)
        return (r * r).sum(1) / self.n

    def learn_exact(self, theta, t):
        return theta - self.pinv[t] @ (self.X[t] @ theta - self.y[t])

    def learn_partial(self, theta, t, k, step_scale=1.0):
        """k GD steps on L_t, step size eta = step_scale * n / lambda_max(X_t X_t^T).
        grad L_t = (2/n) X_t^T (X_t theta - y_t). step_scale = 1 is the SPEC2 definition."""
        eta = step_scale * self.n / self.lam_max[t].item()
        X, y = self.X[t], self.y[t]
        for _ in range(k):
            theta = theta - eta * (2.0 / self.n) * (X.T @ (X @ theta - y))
        return theta

    def one_cycle_operator(self):
        """M = P_{T-1} ... P_0,  P_t = I - X_t^+ X_t."""
        M = torch.eye(self.d)
        for t in range(self.T):
            M = M - self.pinv[t] @ (self.X[t] @ M)
        return M


def schedule_orders(schedule, T, E, seed):
    """List of E lists of task indices."""
    if schedule == "shuffle":
        g = torch.Generator().manual_seed(10_000 + seed)
        return [torch.randperm(T, generator=g).tolist() for _ in range(E)]
    return [list(range(T)) for _ in range(E)]


def run_schedule(rm, schedule, E, seed, k=None, step_scale=1.0, keep_thetas=False):
    """schedule in {'cyclic','shuffle','partial'}. Returns (Ltab (E*T, T) numpy, orders, thetas).
    Ltab[i, s] = L_s immediately after global step i (0-indexed)."""
    T = rm.T
    orders = schedule_orders(schedule, T, E, seed)
    theta = torch.zeros(rm.d)
    Ltab = np.empty((E * T, T))
    thetas = []
    i = 0
    for e in range(E):
        for t in orders[e]:
            if schedule == "partial":
                theta = rm.learn_partial(theta, t, k, step_scale)
            else:
                theta = rm.learn_exact(theta, t)
            Ltab[i] = rm.losses(theta).numpy()
            if keep_thetas:
                thetas.append(theta.clone())
            i += 1
    return Ltab, orders, thetas


def window(Ltab, T, e, s):
    """W(e,s), e 1-indexed >= 2: L_s[tau(e-1,s)], ..., L_s[tau(e,s)-1]; tau(e,s) = (e-1)T + s."""
    assert e >= 2
    t0 = (e - 2) * T + s
    t1 = (e - 1) * T + s
    W = Ltab[t0:t1, s]
    assert len(W) == T
    return W


def window_measures(Ltab, T, E):
    """Returns arrays rho[e-2, s], jstar[e-2, s] for e = 2..E (cyclic-order schedules)."""
    rho = np.empty((E - 1, T))
    js = np.empty((E - 1, T), dtype=int)
    for e in range(2, E + 1):
        for s in range(T):
            W = window(Ltab, T, e, s)
            j = int(np.argmax(W))   # first maximiser on ties
            js[e - 2, s] = j
            rho[e - 2, s] = W[T - 1] / W[j]
    return rho, js


def prequential(Ltab, orders, T, E):
    """Values of L_s just before task s is learned, for epochs 2..E (all tasks)."""
    vals = []
    for e in range(2, E + 1):
        for p, s in enumerate(orders[e - 1]):
            i = (e - 1) * T + p
            vals.append(Ltab[i - 1, s])
    return np.array(vals)


def eigen_summary(M, tol=1e-9):
    lam = np.linalg.eigvals(M.numpy())
    nz = lam[np.abs(lam) > tol]
    nr = lam[np.abs(lam.imag) > tol]
    args = np.abs(np.angle(nr))
    absl = np.abs(lam)
    unit = int(np.sum(np.abs(absl - 1.0) < 1e-9))
    nonunit_nz = absl[(absl > tol) & (np.abs(absl - 1.0) >= 1e-9)]
    return {"n_eig": int(len(lam)), "n_nonzero": int(len(nz)), "n_nonreal": int(len(nr)),
            "median_abs_arg": float(np.median(args)) if len(args) else float("nan"),
            "max_abs_arg": float(np.max(args)) if len(args) else float("nan"),
            # extra diagnostics (not in spec)
            "n_unit_modulus": unit,
            "max_modulus_excl_unit": float(nonunit_nz.max()) if len(nonunit_nz) else float("nan"),
            "n_nonzero_1e-6": int(np.sum(absl > 1e-6)),
            "rank_M": int(np.linalg.matrix_rank(M.numpy()))}
