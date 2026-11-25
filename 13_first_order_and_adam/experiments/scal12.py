"""Scalar single-mode model: f = sum_j w_j e_j  (w_j = share of the journey paid by gradient evaluation j, e_j = output
position where that gradient was evaluated). Checks closed forms and the look-ahead idea. Pure Python."""
def sim(c, mu=0.0, gamma=0.0, mid=False, tol=1e-13, cap=200000):
    x = xp = u = 0.0; f = 0.0; wsum = 0.0; k = 0
    while k < cap:
        y = x + gamma * (x - xp)
        if mid: y = x + 0.5 * c * (1 - x)
        pay = c * (1 - y); w = pay / (1 - mu)
        f += w * y; wsum += w
        u = mu * u + pay; xp = x; x = x + u; k += 1
        if abs(1 - x) < tol and abs(u) < tol: break
        if abs(x) > 1e6: return float("nan"), k, float("nan")
    return f, k, wsum
gd = lambda c: (1 - c) / (2 - c)
gdmid = lambda c: 0.5 + c * c / (2 * (c * c - 2 * c + 4))
hb = lambda c, mu: 1 - ((1 - mu) ** 2 * (1 + mu) + 2 * mu * c) / ((1 - mu) ** 2 * (2 * (1 + mu) - c))
a_hb = lambda c, mu: c * (1 + mu) ** 2 / ((1 - mu) ** 2 * (2 * (1 + mu) - c))
print("GD      c    f_sim     f_closed")
for c in (0.1, 0.5, 1.0, 1.5): print("   %5.2f %+.6f %+.6f" % (c, sim(c)[0], gd(c)))
print("GDmid   c    f_sim     f_closed")
for c in (0.1, 0.5, 1.0, 1.5): print("   %5.2f %+.6f %+.6f" % (c, sim(c, mid=True)[0], gdmid(c)))
print("HB   c  mu   f_sim      f_closed    a_sim     a_closed  steps")
for c, mu in ((0.02, 0.9), (0.1, 0.9), (0.5, 0.9), (0.02, 0.5), (0.5, 0.5), (0.01, 0.9)):
    f, k, ws = sim(c, mu); print("   %5.3f %.1f %+.6f %+.6f %+.5f %+.5f %d (sum w %.6f)" % (c, mu, f, hb(c, mu), 1 - 2 * f, a_hb(c, mu), k, ws))
print("LOOK-AHEAD: a = 1 - 2f for gamma = g* = (1+mu)/(2(1-mu)) and neighbours")
for mu in (0.0, 0.5, 0.9):
    gs = (1 + mu) / (2 * (1 - mu))
    for c in (0.01, 0.02, 0.1, 0.3, 0.5, 1.0):
        row = []
        for g in (0.0, mu, 0.5 * gs, gs, 1.5 * gs):
            f, k, ws = sim(c, mu, g); row.append("%+.5f" % (1 - 2 * f) if f == f else "   nan  ")
        print("   mu %.1f c %5.3f (c_eff %6.3f) a[g=0] %s a[g=mu,Nesterov] %s a[g*/2] %s a[g*=%.2f] %s a[1.5g*] %s" % (mu, c, c / (1 - mu), row[0], row[1], row[2], gs, row[3], row[4]))
