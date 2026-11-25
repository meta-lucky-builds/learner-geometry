"""Pre-check: alignment fraction R = (lam1(L)-lam1(GN))/(lam1(NP)-lam1(GN)) for momentum and Adam,
with GN and NP solved to the learner's ACTUAL final outputs (same fibre). Read-only use of tcr*.py."""
import sys
sys.path.insert(0, ".")
import tcr4  # noqa
import numpy as np, torch
import tcr6, tcr7
from tcr4 import fo_learn_C, np_model

MU = tcr4.MOM_MU


def f_gd(c):
    return (1 - c) / (2 - c)


def f_hb(c, mu=MU):
    return 1 - ((1 - mu) ** 2 * (1 + mu) + 2 * mu * c) / ((1 - mu) ** 2 * (2 * (1 + mu) - c))


h, seed = int(sys.argv[1]), int(sys.argv[2])
S = tcr6.setup(h, seed)
net = tcr7.Net(S.model, S.X)
npm = np_model(S.model, S.X)
lam = S.lam.numpy()
l0 = tcr6.lam1(S.model, S.theta_A, S.X)
print("h=%d seed=%d mu=%.2f lam1=%.3f ADAM_LR=%s" % (h, seed, MU, lam[0], getattr(tcr4, "ADAM_LR", None)))
g = torch.Generator().manual_seed(555)
dirs = {"q1": S.Q[:, 0], "q2": S.Q[:, 1], "q4": S.Q[:, 3], "q10": S.Q[:, 9]}
d = torch.randn(tcr6.N, generator=g); dirs["rand"] = d / torch.linalg.norm(d)
s = 0.05
cases = [("GD", 0.5), ("GD", 1.5), ("MOM", 0.002), ("MOM", 0.02), ("MOM", 0.2), ("MOM", 1.0), ("ADAM", None)]
for dn, e in dirs.items():
    idx = {"q1": 0, "q2": 1, "q4": 3, "q10": 9}.get(dn)
    for kind, c in cases:
        num = den = 0.0; numt = dent = 0.0; st = []
        for sg in (1, -1):
            y = (S.yA + sg * s * e)
            C = np.zeros((tcr6.N, tcr6.N))
            th, C, info = fo_learn_C(npm, S.model, S.theta_A, S.X, y.numpy(), S.F_A_np, C, kind, c)
            st.append("%s/%d" % (info["status"][:4], info["steps"]))
            F_end = S.model.f(th, S.X)
            thg, _, _, _, ig = tcr7.gn_run(S.model, S.theta_A, S.X, F_end)
            thn, inn = tcr7.np_solve(net, S.theta_A, F_end)
            if thg is None or thn is None:
                st.append("REF-FAIL"); continue
            lL = tcr6.lam1(S.model, th, S.X); lG = tcr6.lam1(S.model, thg, S.X); lN = net.lam1(thn)
            num += lL - lG; den += lN - lG
            tL = tcr6.K_of(S.model, th, S.X).trace().item(); tG = tcr6.K_of(S.model, thg, S.X).trace().item(); tN = float(np.trace(net.K(thn)))
            numt += tL - tG; dent += tN - tG
        if idx is not None and kind != "ADAM":
            ci = c * lam[idx] / lam[0]
            pred = "%.3f" % (f_gd(ci) if kind == "GD" else f_hb(ci))
        else:
            pred = "  -  "
        print("  %-5s %-4s c=%-6s R(lam1)=%8.3f  R(trK)=%8.3f  pred=%s   [%s]" % (dn, kind, c, num / den if den else float("nan"), numt / dent if dent else float("nan"), pred, " ".join(st)))
