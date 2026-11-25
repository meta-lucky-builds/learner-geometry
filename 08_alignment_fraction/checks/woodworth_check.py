"""Independent analytic-model check of the 'flow lands halfway' statement.
Model (Woodworth et al. 2002.09277, one coordinate): beta = u^2 - v^2, start u = v = alpha.
Fibre of target b: hyperbola u^2 - v^2 = b.  Tangent kernel K = |grad beta|^2 = 4(u^2+v^2).
 flow endpoint : gradient flow conserves u*v = alpha^2 (exact).
 NP            : nearest point of the hyperbola to (alpha, alpha).
 GN            : iterated min-norm Gauss-Newton from the start (first step is the tangent-space min-norm step).
Ratio R = (K_flow - K_GN)/(K_NP - K_GN); prediction 1/2 as b/alpha^2 -> 0.
Also damped GN with step fraction c: prediction (1-c)/(2-c); GD with step factor c: (1-c)/(2-c)."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
al = 1.0
K = lambda u, v: 4 * (u * u + v * v)
def flow_end(b):
    # u v = al^2, u^2 - v^2 = b
    u2 = (b + np.sqrt(b * b + 4 * al ** 4)) / 2
    u = np.sqrt(u2); return u, al * al / u
def np_end(b):
    # parametrise hyperbola by v: u = sqrt(b + v^2)
    f = lambda v: (np.sqrt(b + v * v) - al) ** 2 + (v - al) ** 2
    r = minimize_scalar(f, bounds=(max(0.0, al - 1.0), al + 1.0), method="bounded", options={"xatol": 1e-14})
    v = r.x; return np.sqrt(b + v * v), v
def gn_iter(b, c=1.0, tol=1e-14, u=al, v=al):
    for _ in range(100000):
        r = u * u - v * v - b
        if abs(r) < tol: break
        g = np.array([2 * u, -2 * v]); step = -c * r * g / (g @ g)
        u, v = u + step[0], v + step[1]
    return u, v
def gd(b, c):
    u, v = al, al
    eta = c / K(al, al)          # step factor c relative to the kernel at the start (n = 1, loss r^2/2)
    for _ in range(2000000):
        r = u * u - v * v - b
        if abs(r) < 1e-14: break
        u, v = u - eta * r * 2 * u, v + eta * r * 2 * v
    return u, v
print("b/alpha^2 | R_flow (pred 0.5) | damped GN c=0.5 (pred %.4f) c=0.1 (pred %.4f) | GD c=0.5 (pred %.4f) c=1.5 (pred %.4f) c=1.9 (pred %.4f)" % ((1-.5)/(2-.5), (1-.1)/(2-.1), (1-.5)/(2-.5), (1-1.5)/(2-1.5), (1-1.9)/(2-1.9)))
for b in (0.4, 0.2, 0.1, 0.05, 0.02):
    kg = K(*gn_iter(b)); kn = K(*np_end(b)); den = kn - kg
    R = lambda p: (K(*p) - kg) / den
    print("  %5.2f   | %.4f | %.4f %.4f | %.4f %.4f %.4f   [K_GN %.6f K_flow %.6f K_NP %.6f]" % (b, R(flow_end(b)), R(gn_iter(b, 0.5)), R(gn_iter(b, 0.1)), R(gd(b, 0.5)), R(gd(b, 1.5)), R(gd(b, 1.9)), kg, K(*flow_end(b)), kn))
