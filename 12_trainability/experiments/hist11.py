"""History test (FINDINGS.md sec. 43): structured vs shuffled earlier tasks, explicit vs two-gradient learner, then the true labels."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, time, math, json, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", int(sys.argv[2]), OUT + "/hist11_out")
cname = sys.argv[1]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; td = combo["test"]()
lab = ctx.labels.cpu(); C = 10; n = lab.numel()
coarse = [lab % 2, lab // 5, lab % 3, lab // 2]
oh = lambda l: torch.nn.functional.one_hot(l.long(), C).to(L.DT).reshape(-1).to(st.F0.device)
Wv = [combo["dirs"][x] for x in S.W_NAMES]
def em(prob, st_, t):
    h = L.jump(prob, st_.th0, st_.F0 + 0.5 * (t - st_.F0), Jb=st_.J0)
    return L.jump_a(prob, st_.th0, t, Jb=prob.J(h["theta"]))["theta"]
def fit(th, Y, learner):
    stc = L.Start(prob, th)
    if learner == "EMS(1)": return em(prob, stc, Y)
    r = L.run_learner9(prob, stc, Y, L.vnorm(Y - stc.F0), L.L_SL(2) if learner == "SL(2)" else L.L_SL(8)); assert r["status"] == "CONVERGED", r["status"]; return r["theta"]
def final(th):
    thf = fit(th, combo["Y"], "SL(8)"); loss, acc = L.test_loss_acc(L.test_outputs(prob, td["X"], thf), td)
    K = S.kernel(L, prob, th); q = S.q_from_kernel(L, K, Wv); k = S.ksum(K); del K
    return loss, acc, sum(math.log(x) for x in q) / 8, k["A"], k["lamN"], L.vnorm(th - st.th0)
print("%s fresh network            : final test loss %.5f acc %.4f | before final task: Q %.3f A %.3f lamN %.3f dist %.3f" % ((cname,) + final(st.th0)), flush=True)
hist = [("structured", coarse)] + [("shuffled-%d" % p, [c[torch.randperm(n, generator=L.gen(4000 + p))] for c in coarse]) for p in range(3)]
for hname, tasks in hist:
    for learner in ("SL(2)", "EMS(1)"):
        t0 = time.time(); th = st.th0
        for l in tasks:
            th = fit(th, oh(l), learner); assert L.vnorm(prob.F(th) - oh(l)) <= 1e-6 * 10
            L.free_memory()
        print("%s %-11s history, %-6s: final test loss %.5f acc %.4f | before final task: Q %.3f A %.3f lamN %.3f dist %.3f (%.0fs)" % ((cname, hname, learner) + final(th) + (time.time() - t0,)), flush=True)
print("hist11 done", flush=True)
