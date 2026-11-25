"""Width sweep for ReLU (prediction written before the run, FINDINGS.md sec. 28): at a fixed relative move size the deviation of
ReLU learners from the smooth fractions falls as the network gets wider (more, smaller switches), roughly like width^-1/2,
while the number of switches along the path rises. Local, synthetic inputs, n = 8, 196-w-w-10 ReLU MLP at initialisation,
direction 'label', library reference path (no nearest-point solve). rho_PP = <v_L, v_PP>/|v_PP|^2."""
import sys, time, statistics as st_
sys.path.insert(0, sys.argv[1])
from common9 import load
L, CFG = load("tiny")
import torch
torch.set_num_threads(4)
LEARN = [(L.L_SL(2), 0.25), (L.L_SL(8), 0.4375), (L.L_SL(32), 0.484375), (L.L_DJ(0.5), 1 / 3)]
widths = [int(w) for w in sys.argv[2].split(",")]
sizes = [0.01, 0.03]
print("width seed size sign | n_sw(1) max|om-tau| max om_perp | rho_PP for SL(2) SL(8) SL(32) DJ(0.5) | median |rho-f| (sec)", flush=True)
summ = {}
for w in widths:
    for seed in ((0, 1) if w <= 256 else (0,)):
        X, labels, Y = L.get_data(seed, 8)
        prob, th = L.layered_problem(X, [196, w, w, CFG["N_CLASSES"]], "relu", seed, name="relu-w%d" % w)
        stt = L.Start(prob, th)
        dirs, dinfo, S0 = L.make_directions(stt, seed, 8)
        for fr in sizes:
            for sg in (1, -1):
                t0 = time.time()
                d = sg * dirs["label"]; s = fr * S0; t = stt.F0 + s * d
                try:
                    ref = L.RefPath(prob, stt, d, s, 64)
                    vPP, om, omp = L.R_profile(ref)
                    n2 = float(vPP @ vPP)
                    rhos = []
                    for Ld, f in LEARN:
                        res = L.run_learner9(prob, stt, t, s, Ld)
                        vL = stt.V0(res["theta"] - stt.th0)
                        rhos.append(float(vL @ vPP) / n2)
                    dev = [abs(r - f) for r, (_, f) in zip(rhos, LEARN)]
                    md = st_.median(dev)
                    summ.setdefault((w, fr), []).append((md, ref.nsw[32], max(abs(om[k] - k / 16) for k in range(17)), max(omp[:17])))
                    print("%5d %d %.2f %+d | %5d  %.3f  %.3f | %s | %.4f (%.0f)" % (w, seed, fr, sg, ref.nsw[32], summ[(w, fr)][-1][2],
                          summ[(w, fr)][-1][3], " ".join("%.3f" % r for r in rhos), md, time.time() - t0), flush=True)
                except Exception as e:
                    print("%5d %d %.2f %+d | FAILED %s: %s" % (w, seed, fr, sg, type(e).__name__, str(e)[:80]), flush=True)
        del prob, stt
        L.free_memory()
    for fr in sizes:
        v = summ.get((w, fr), [])
        if v:
            print("   SUMMARY width %d size %.2f: cases %d  median switches %.0f  median max|om-tau| %.3f  median om_perp %.3f  median of median |rho-f| %.4f" % (
                w, fr, len(v), st_.median(x[1] for x in v), st_.median(x[2] for x in v), st_.median(x[3] for x in v), st_.median(x[0] for x in v)), flush=True)
