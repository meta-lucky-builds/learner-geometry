"""Heavy-ball momentum and Adam in the u^2 - v^2 model: measured alignment fraction R vs the closed form
f_HB(c, mu) = 1 - [(1-mu)^2 (1+mu) + 2 mu c] / [(1-mu)^2 (2(1+mu) - c)],  c = eta * K.
GN and NP are solved to the learner's actual final output (same fibre)."""
import numpy as np
from scipy.optimize import minimize_scalar
al = 1.0
K = lambda u, v: 4 * (u * u + v * v)
def np_end(b):
    f = lambda v: (np.sqrt(b + v * v) - al) ** 2 + (v - al) ** 2
    v = minimize_scalar(f, bounds=(al - 0.9, al + 0.9), method="bounded", options={"xatol": 1e-15}).x
    return np.sqrt(b + v * v), v
def gn(b):
    u, v = al, al
    for _ in range(1000):
        r = u * u - v * v - b
        if abs(r) < 1e-15: break
        g = np.array([2 * u, -2 * v]); s = -r * g / (g @ g); u, v = u + s[0], v + s[1]
    return u, v
def hb(b, c, mu):
    u, v = al, al; pu = pv = 0.0; eta = c / K(al, al)
    for _ in range(5000000):
        r = u * u - v * v - b
        if abs(r) < 1e-15 and abs(pu) + abs(pv) < 1e-15: break
        pu = mu * pu - eta * r * 2 * u; pv = mu * pv + eta * r * 2 * v
        u, v = u + pu, v + pv
    return u, v
def adam(b, lr, b1=0.9, b2=0.999, eps=1e-8, steps=200000):
    u, v = al, al; m = np.zeros(2); s = np.zeros(2)
    for t in range(1, steps + 1):
        r = u * u - v * v - b
        g = r * np.array([2 * u, -2 * v])
        m = b1 * m + (1 - b1) * g; s = b2 * s + (1 - b2) * g * g
        d = lr * (m / (1 - b1 ** t)) / (np.sqrt(s / (1 - b2 ** t)) + eps)
        u, v = u - d[0], v - d[1]
    return u, v
def signgd(b, lr, steps=4000):
    u, v = al, al
    for t in range(steps):
        r = u * u - v * v - b
        g = r * np.array([2 * u, -2 * v]); d = lr * np.sign(g) * min(1.0, abs(r) / (lr * 4))
        u, v = u - d[0], v - d[1]
    return u, v
def R(p):
    b = p[0] ** 2 - p[1] ** 2
    kg = K(*gn(b)); kn = K(*np_end(b))
    return (K(*p) - kg) / (kn - kg), b
f_hb = lambda c, mu: 1 - ((1 - mu) ** 2 * (1 + mu) + 2 * mu * c) / ((1 - mu) ** 2 * (2 * (1 + mu) - c))
b = 0.02
print("heavy ball, target b = %.2f:  mu  c | measured R | closed form" % b)
for mu in (0.5, 0.9):
    for c in (0.001, 0.0105, 0.05, 0.3, 1.0, 2.0, 3.0):
        if c >= 2 * (1 + mu): continue
        r, bb = R(hb(b, c, mu))
        print("   mu=%.1f c=%-6g  R=%10.4f   f_HB=%10.4f" % (mu, c, r, f_hb(c, mu)))
print("zero crossing c* = (1-mu)^2 (1+mu)/(1+mu^2): mu=0.9 -> %.5f ; in effective step units c*/(1-mu) = %.4f" % (0.01 * 1.9 / 1.81, 0.1 * 1.9 / 1.81))
print("\nAdam (lr), same model; R measured on Adam's own final fibre:")
for lr in (1e-4, 1e-3, 1e-2):
    p = adam(b, lr); r, bb = R(p)
    print("   lr=%g  final output %.6f (target %.2f)  R=%.4f   final (u-1, v-1) = (%.5f, %.5f)" % (lr, bb, b, r, p[0] - 1, p[1] - 1))
print("\nsign-like descent (equal step on both weights):")
for lr in (1e-4, 1e-3):
    p = signgd(b, lr); r, bb = R(p)
    print("   lr=%g  final output %.6f  R=%.4f" % (lr, bb, r))
