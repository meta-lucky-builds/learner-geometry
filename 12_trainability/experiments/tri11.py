"""Triangle loops in both orientations: even part (heat) vs odd part (area). Prediction in FINDINGS.md sec. 43."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", int(sys.argv[2]), OUT + "/tri11_out")
cname = sys.argv[1]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; s = combo["sizes"]["s1"]
u, w = combo["dirs"]["E1"], combo["dirs"]["E2"]; d3 = w - u; n3 = L.vnorm(d3); e3 = d3 / n3
def vnpe(x, size):
    r = {}
    for sg in (1, -1):
        rec, v = S.reference(L, prob, st, x, size, sg); r[sg] = v
    return S.even(r[1], r[-1])
Vsum = vnpe(u, s) + vnpe(w, s) + vnpe(e3, s * n3); nV = L.vnorm(Vsum)
print("%s |Vsum| %.4e  (legs: s, s, %.3f s)" % (cname, nV, n3), flush=True)
def em(prob, st_, t):
    h = L.jump(prob, st_.th0, st_.F0 + 0.5 * (t - st_.F0), Jb=st_.J0); return L.jump_a(prob, st_.th0, t, Jb=prob.J(h["theta"]))["theta"]
def go(th, t, name):
    stc = L.Start(prob, th)
    if name == "EM": return em(prob, stc, t)
    r = L.run_learner9(prob, stc, t, L.vnorm(t - stc.F0), L.L_SL(2) if name == "SL(2)" else L.L_AJ(1.0)); assert r["status"] == "CONVERGED", r["status"]; return r["theta"]
odd = {}
for name, f in (("EM", 0.5), ("SL(2)", 0.25), ("AJ(1)", 1.0)):
    t0 = time.time(); V = {}
    for tag, a, b in (("fwd", u, w), ("rev", w, u)):
        th = go(st.th0, st.F0 + s * a, name); th = go(th, st.F0 + s * b, name); th = go(th, st.F0, name)
        V[tag] = st.V0(th - st.th0); cl = L.vnorm(prob.F(th) - st.F0) / s; L.free_memory()
    ev = 0.5 * (V["fwd"] + V["rev"]); od = 0.5 * (V["fwd"] - V["rev"]); odd[name] = od
    print("%s %-6s f %.4f | even (heat): rho against Vsum %+.5f (predicted %+.4f) cos %+.4f |even|/|Vsum| %.5f | odd (area): |odd|/|Vsum| %.5f |even|/|odd| %.4f | close %.1e (%.0fs)" % (
        cname, name, f, float(ev @ Vsum) / nV ** 2, f - 0.5, L.cosine(ev, Vsum), L.vnorm(ev) / nV, L.vnorm(od) / nV, L.vnorm(ev) / max(L.vnorm(od), 1e-300), cl, time.time() - t0), flush=True)
ks = list(odd)
for i in range(3):
    for j in range(i):
        print("   area parts %s vs %s: cosine %+.4f, norm ratio %.3f" % (ks[i], ks[j], L.cosine(odd[ks[i]], odd[ks[j]]), L.vnorm(odd[ks[i]]) / L.vnorm(odd[ks[j]])))
print("tri11 done", flush=True)
