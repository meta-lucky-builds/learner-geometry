"""Continue the SPEC11 Part B star on gelu with DJ(1.5) past the rig's 6 pairs (prediction in FINDINGS.md sec. 43)."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import sys, os, json, time, math, torch
B = os.environ.get("TCR_BUNDLE", DATA + "/spec11_bundle")
sys.path.insert(0, B); sys.dont_write_bytecode = True
import spec11_common as S
L, CFG = S.load(B, "cpu", 7, OUT + "/collapse11_out")
lname, cname, npairs = sys.argv[1], sys.argv[2], int(sys.argv[3])
spec = {"DJ(1.5)": ("DJ", 1.5), "DJ(1)": ("DJ", 1.0), "AJ(1)": ("AJ", 1.0)}[lname]
ctx = L.Ctx9(S.COMBOS[cname], B + "/run"); combo = S.combo_from_ctx(L, CFG, ctx); prob, st = combo["prob"], combo["st"]; sd = combo["sd"]
dirs = dict(combo["dirs"])
for m in range(7, npairs + 1):
    dirs["E%d" % m] = S.label_dir(L, st, 8000 + 100 * sd + m) if m % 2 else S.gauss_dir(L, st, 9000 + 100 * sd + m)
s = combo["sizes"]["s1"]; Ld = S.make_learner(L, spec); Wv = [dirs[x] for x in S.W_NAMES]
K0 = S.kernel(L, prob, st.th0); q0 = S.q_from_kernel(L, K0, Wv); k0 = S.ksum(K0); del K0
out = open(OUT + "/collapse11_%s_%s.jsonl" % (cname, lname.replace("(", "_").replace(")", "")), "a")
th = st.th0; f = L.predict_f9(Ld)
STATE = OUT + "/collapse11_state_%s_%s.pt" % (cname, lname.replace("(", "_").replace(")", ""))
m0 = 0
if os.path.exists(STATE):
    stt = torch.load(STATE, map_location="cpu"); th = stt["th"].to(L.DT); m0 = int(stt["m"]); print("resuming after pair", m0, flush=True)
for m in range(m0 + 1, npairs + 1):
    t0 = time.time(); status = "OK"; info = []
    for sigma in S.SIGNS:
        try:
            stc = L.Start(prob, th); leg = S.two_legs(L, prob, stc, st.F0, Ld, dirs["E%d" % m], s, sigma, ""); del stc
            info.append((leg["rec"]["leg1_status"], leg["rec"]["leg2_status"], leg["rec"]["close"], leg["rec"].get("leg1", {}).get("steps"), leg["rec"].get("leg2", {}).get("steps")))
            if leg["rec"]["status"] != "OK": status = leg["rec"]["status"]; break
            th = leg["th2"]
        except Exception as e:
            status = "EXC:" + type(e).__name__; info.append(repr(e)[:120]); break
        L.free_memory()
    rec = dict(m=m, status=status, info=info, sec=time.time() - t0, t=(1 - 2 * f) * 2 * m)
    try:
        K = S.kernel(L, prob, th); k = S.ksum(K); lam = torch.linalg.eigvalsh(K)
        rec.update(lamN=k["lamN"] / k0["lamN"], lam1=k["lam1"] / k0["lam1"], trK=k["trK"] / k0["trK"], A=k["A"], lam_small=[float(x) / k0["lamN"] for x in lam[:4]])
        try:
            q = S.q_from_kernel(L, K, Wv); rec.update(inv_q=sum(a / b for a, b in zip(q0, q)) / 8, q_rel=sum(b / a for a, b in zip(q0, q)) / 8)
        except Exception as e:
            rec["q_error"] = type(e).__name__
        del K
    except Exception as e:
        rec["kernel_error"] = repr(e)[:100]
    out.write(json.dumps(rec) + "\n"); out.flush()
    print("pair %2d t %+6.1f status %-14s lamN/lamN0 %s  four smallest %s  mean q0/q %s  trK %s lam1 %s  %s (%.0fs)" % (
        m, rec["t"], status, ("%.4f" % rec["lamN"]) if "lamN" in rec else "-", " ".join("%.4f" % x for x in rec.get("lam_small", [])),
        ("%.4f" % rec["inv_q"]) if "inv_q" in rec else "-", ("%.4f" % rec["trK"]) if "trK" in rec else "-", ("%.4f" % rec["lam1"]) if "lam1" in rec else "-", info[-1] if info else "", rec["sec"]), flush=True)
    if status != "OK": break
    torch.save(dict(th=th.detach().cpu(), m=m), STATE + ".tmp"); os.replace(STATE + ".tmp", STATE)
print("collapse11 done", flush=True)
