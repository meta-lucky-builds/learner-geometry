"""Corollary check: predictions on UNSEEN inputs. All learners below fit the same training outputs; the law says their
predictions on held-out inputs differ along ONE function g(x), with coefficient equal to the alignment fraction:
   F_test(theta_L) - F_test(theta_TH) = f_L * [ F_test(theta_NP) - F_test(theta_TH) ]   (leading order, even part)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "review8", "final"))
exec(open(os.path.join(HERE, "..", "review8", "final", "config_cell.py"), encoding="utf-8").read())
import part8lib as L
CFG["JUMP_TOL"] = 1e-13; CFG["NP_RES_TOL"] = 1e-12
L.configure(CFG)
import torch
torch.set_default_dtype(torch.float64)
L._load_mnist_raw()
act = sys.argv[1] if len(sys.argv) > 1 else "tanh"
CFG["WIDTH"] = 32; CFG["N_TRAIN"] = 8; L.configure(CFG)
cb = dict(act=act, start="trained", seed=0)
prob, th, Y, labels = L.combo_problem(cb)
th, _ = L.fit_dj(prob, th, Y)
st = L.Start(prob, th)
dirs, dinfo, S0 = L.make_directions(st, 0, 8)
Xtest = torch.randn(200, 196, generator=L.gen(999))
ftest = lambda t: L.mlp_forward(t, Xtest, prob.shapes, prob.act)[0]
gd = lambda c: L.L_GD(c, 1e-6, 300000, dir_stop_rel=1e-6, dir_window=100)
LS = [("SL(2)", L.L_SL(2), 0.25), ("SL(32)", L.L_SL(32), 0.484375), ("DJ(0.5)", L.L_DJ(0.5), 1 / 3), ("DJ(1.5)", L.L_DJ(1.5), -1.0), ("DJ(1.9)", L.L_DJ(1.9), -9.0),
      ("PROX", L.L_PROX(1e-2), 1.0)]
print("%s MLP 196-32-32-10, 8 training inputs, 200 held-out inputs (2000 held-out outputs)" % act)
for dn in ("eigA", "label"):
    d = dirs[dn]; s = 0.01 * S0
    print(" direction %s: learner | predicted f | measured coefficient on held-out outputs | cosine | size of the spread relative to the held-out change itself" % dn)
    for nm, Ld, f in LS:
        num = den = 0.0; dL_all = []; dN_all = []; base = []
        for sg in (1, -1):
            res = L.run_learner(prob, st, st.F0 + sg * s * d, s, Ld)
            thL = res["theta"]; FL = prob.F(thL)
            thN = L.nearest_point(prob, st.th0, FL, thL)["theta"]
            thT = L.jump(prob, st.th0, FL, Jb=st.J0)["theta"]
            dL_all.append(ftest(thL) - ftest(thT)); dN_all.append(ftest(thN) - ftest(thT)); base.append(ftest(thT) - ftest(st.th0))
        a = 0.5 * (dL_all[0] + dL_all[1]); b = 0.5 * (dN_all[0] + dN_all[1])
        coef = float(a @ b) / float(b @ b); cos = float(a @ b) / (L.vnorm(a) * L.vnorm(b))
        print("   %-8s  f = %+.4f   coefficient = %+.4f   cos = %+.5f   |NP - TH| / |TH - start| on held-out = %.3e" % (nm, f, coef, cos, L.vnorm(b) / L.vnorm(base[0])))
