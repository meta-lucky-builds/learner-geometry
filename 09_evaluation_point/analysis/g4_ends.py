"""Follow-up to G4 of the SPEC9 run: why does the order of the learners by test loss turn round at large moves?

RUN ONLY AFTER THE MAIN RUN HAS FINISHED: this script uses a GPU and re-runs learners. It refuses to start on a busy GPU.
It writes nothing into the run directory.

It repeats four of the G4 learners with the library's own code, DJ(1) (f = 0), SL(4) (0.375), DJ(0.25) (0.4286) and
DJ(1.5) (-1), for every G4 combination and every tau, and this time keeps the held-out outputs F_test of each end point.
'repro' compares the repeated run with the stored G4 record (test loss and weight distance; expected about 1e-10 or less).

Per end point (Fv = F_test minus its per-class mean over the test inputs, Y = one-hot test labels):
  a*    = <Y, Fv>/|Fv|^2 : the factor by which the centred function would have to be multiplied to fit the test labels
          best. a* > 1: the function is too SMALL on the test inputs; a* < 1: too large.
  loss* = test loss after that one-number rescaling. If the learners differ in 'loss' but not in 'loss*', the whole
          difference between them is the size of the function. If they differ in 'loss*' too, it is more than size.
Per learner against the DJ(1) end point 'lo' of the same block (D = F_test(learner) - F_test(lo), r_lo = F_lo - Y):
  exact:  loss - loss_lo = <r_lo, D>/N + |D|^2/(2N) = const + dN + dS + drest + quad
  D minus its per-class mean is fitted by least squares as eN Nv + eS Sv + rest, with Nv = the centred function at the
  start (theta0, random initialisation) and Sv = the centred change the exact jump made (F_lo - F_test(theta0)).
  eN, eS = relative growth of each part; dN = eN gN, dS = eS gS, gN = <r_lo, Nv>/N, gS = <r_lo, Sv>/N (test-loss change per
  unit growth of each part; gS < 0 means the part that was learned is too small on the test inputs).
  R2 = share of D (centred) that is a rescaling of the two parts.

What was written down BEFORE this was run (FINDINGS.md sec. 23), 'size' reading: eN > 0 and eS > 0 for f > 0 (both parts grow
with f); gN > 0 everywhere; where the G4 order turns round, gS < 0 and |dS| > dN, and 'loss*' no longer prefers larger f.
If instead drest carries the turn-round, or loss* (and accuracy) improve with f, the effect is more than size.

usage on the rig, one GPU:        python g4_ends.py
four GPUs (after the main run):   for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k python g4_ends.py $k 4 > g4_ends_$k.txt 2>&1 & done; wait
local test: PART9_NB=<notebook> PART9_MODE=tiny PART9LIB_DIR=... PART9_RUN_DIR=... PART9_NO_DOWNLOAD=1 PART9_DEVICE=cpu python g4_ends.py
"""
import os, sys, json, re, time

nbp = os.path.abspath(os.environ.get("PART9_NB", "/workspace/evaluation_point_tests.ipynb"))
mode = os.environ.get("PART9_MODE")
k_w = int(sys.argv[1]) if len(sys.argv) > 1 else 0
m_w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
assert 0 <= k_w < m_w
os.chdir(os.path.dirname(nbp))
sys.path.insert(0, os.environ.get("PART9LIB_DIR", os.getcwd()))
nb = json.load(open(nbp, encoding="utf-8"))
src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "S1_FRAC, S2_FRAC" in "".join(c["source"])][0]
if mode:
    src, k = re.subn(r'MODE = "\w+"', 'MODE = "%s"' % mode, src, count=1)
    assert k == 1
g = {}
exec(src, g)
CFG = g["CFG"]
if os.environ.get("PART9_NO_DOWNLOAD"):
    CFG["MNIST_DOWNLOAD"] = False

import torch
import part9lib as L
L.configure(CFG)
dev = os.environ.get("PART9_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if dev != "cpu":
    free, tot = torch.cuda.mem_get_info()
    if free < 0.8 * tot and not os.environ.get("G4_ENDS_FORCE"):
        sys.exit("g4_ends: this GPU is in use (%.1f of %.1f GB free). Wait until the main run has finished, then start again."
                 % (free / 1e9, tot / 1e9))
L.set_device(dev)
L._load_mnist_raw()
rd = os.environ.get("PART9_RUN_DIR", CFG["RUN_DIR"])
C = CFG["N_CLASSES"]
SPECS = [("DJ", 1.0), ("SL", 4), ("DJ", 0.25), ("DJ", 1.5)]          # the first one is the reference 'lo' (f = 0)
FNOM = {}
for x, fnom in CFG["G4_LEARNERS"]:
    for sp in SPECS:
        if x[0] == sp[0] and len(x) == len(sp) and all(float(a) == float(b) for a, b in zip(x[1:], sp[1:])):
            FNOM[sp] = fnom
assert all(sp in FNOM for sp in SPECS), "a learner of this script is not one of the G4 learners of the run"
assert FNOM[SPECS[0]] == 0.0
print("g4_ends: MODE %s data_source %s device %s run_dir %s worker %d of %d" % (
    CFG["MODE"], L.DATA_SOURCE, dev, os.path.abspath(rd), k_w, m_w), flush=True)


def rec(name):
    p = os.path.join(rd, "done", L.safe_name(name) + ".json")
    return L.read_json(p) if os.path.exists(p) else None


def case_name(cn, Ld, tau, fnom):
    return L._case9(cn, "G4", "G4", Ld, "true", f"tau{tau:g}", tau, 1, f_nominal=fnom)["name"]


def centre(x):
    X = x.reshape(-1, C)
    m = X.mean(0, keepdim=True)
    return X - m, m.expand_as(X)


def dot(a, b):
    return float((a * b).sum())


def cos(a, b):
    return dot(a, b) / max((dot(a, a) * dot(b, b)) ** 0.5, 1e-300)


def pick(M, y):
    return float(M.gather(1, y.view(-1, 1)).sum())


cbs = sorted(L.combos("G4"), key=lambda cb: -int(cb["n"]))            # stable sort: the larger training sets first
mine = [cb for i, cb in enumerate(cbs) if i % m_w == k_w]
eta = 0.0
for cb in mine:
    for tau in CFG["G4_TAUS"]:
        for sp in SPECS:
            r = rec(case_name(L.cname9(cb), L.L_from_spec(sp), tau, FNOM[sp]))
            eta += float(r.get("wall_s", 0.0)) if r else 0.0
print("g4_ends: combinations %s; the same cases took %.0f min in the main run" % ([L.cname9(cb) for cb in mine], eta / 60), flush=True)

bundle = dict(mode=CFG["MODE"], data_source=L.DATA_SOURCE, combos={})
for cb in mine:
    cn = L.cname9(cb)
    t0 = time.time()
    ctx = L.Ctx9(cb, rd)
    prob, st = ctx.prob, ctx.st
    td = ctx.test()
    y = td["labels"].long()
    Yt = td["Y"]
    F0t, l0, a0 = ctx.test0()
    N = F0t.numel()
    Nv, _ = centre(F0t)
    entry = dict(labels=y.cpu().to(torch.int16), F0=F0t.reshape(-1, C).float().cpu(), ends={})
    for tau in CFG["G4_TAUS"]:
        t = st.F0 + tau * (ctx.Y - st.F0)
        s = L.vnorm(t - st.F0)
        ends = []
        for sp in SPECS:
            Ld = L.L_from_spec(sp)
            ln = L.learner_name9(Ld)
            try:
                res = L.run_learner9(prob, st, t, s, Ld)
                thL = res["theta"]
                if not bool(torch.isfinite(thL).all()):
                    raise RuntimeError("non-finite weights")
                F = L.test_outputs(prob, td["X"], thL)
                loss, acc = L.test_loss_acc(F, td)
                tr, dist = L.vnorm(prob.F(thL) - t) / s, L.vnorm(thL - st.th0)
                r = rec(case_name(cn, Ld, tau, FNOM[sp]))
                rep = (max(abs(loss - r["loss_L"]) / abs(r["loss_L"]), abs(dist - r["dist"]) / max(abs(r["dist"]), 1e-300))
                       if r and "loss_L" in r and "dist" in r else float("nan"))
                Fv, _ = centre(F)
                a_star = pick(Fv, y) / dot(Fv, Fv)
                rs = (a_star * Fv + (F.reshape(-1, C) - Fv)) - Yt.reshape(-1, C)
                ends.append(dict(ln=ln, f=FNOM[sp], status=res["status"], F=F, loss=loss, acc=acc, tr=tr, dist=dist, rep=rep,
                                 a_star=a_star, loss_star=dot(rs, rs) / (2.0 * N)))
                entry["ends"][(tau, ln)] = F.reshape(-1, C).float().cpu()
            except Exception as e:
                print("%-19s tau=%-5g %-9s FAILED here: %s: %s" % (cn, tau, ln, type(e).__name__, str(e)[:120]), flush=True)
            L.free_memory()
        if not ends or ends[0]["ln"] != L.learner_name9(L.L_from_spec(SPECS[0])):
            print("%-19s tau=%-5g no reference end point; block skipped" % (cn, tau), flush=True)
            continue
        lo = ends[0]
        Flo = lo["F"].reshape(-1, C)
        r_lo = Flo - Yt.reshape(-1, C)
        Sv = centre(lo["F"])[0] - Nv
        a11, a12, a22 = dot(Nv, Nv), dot(Nv, Sv), dot(Sv, Sv)
        det = a11 * a22 - a12 * a12
        gN, gS = dot(r_lo, Nv) / N, dot(r_lo, Sv) / N
        print("\n%s tau=%g   test loss at theta0 %.6f, accuracy %.4f | at the DJ(1) end point: gN %+.3e  gS %+.3e  cos(Nv,Sv) %+.3f  "
              "|Sv|/|Nv| %.3f  best factor for the learned part alone <Y,Sv>/|Sv|^2 = %.3f" % (
                  cn, tau, l0, a0, gN, gS, cos(Nv, Sv), (a22 / a11) ** 0.5, pick(Sv, y) / a22), flush=True)
        print("   %-9s %7s %-10s %9s %9s %9s %7s %7s %9s | %10s = %10s %10s %10s %10s %10s | %10s %10s %6s" % (
            "learner", "f_nom", "status", "train res", "repro", "loss", "acc", "a*", "loss*", "loss-lo", "const", "dN", "dS",
            "drest", "quad", "eN", "eS", "R2"))
        for en in ends:
            head = "   %-9s %+7.4f %-10s %9.2e %9.1e %9.6f %7.4f %7.3f %9.6f" % (
                en["ln"], en["f"], en["status"], en["tr"], en["rep"], en["loss"], en["acc"], en["a_star"], en["loss_star"])
            if en is lo:
                print(head + " | (reference)", flush=True)
                continue
            D = en["F"].reshape(-1, C) - Flo
            Dv, Dc = centre(D)
            g1, g2 = dot(Nv, Dv), dot(Sv, Dv)
            eN = (g1 * a22 - g2 * a12) / det
            eS = (g2 * a11 - g1 * a12) / det
            rest = Dv - eN * Nv - eS * Sv
            const, dN, dS, dr, quad = dot(r_lo, Dc) / N, eN * gN, eS * gS, dot(r_lo, rest) / N, dot(D, D) / (2.0 * N)
            dl = en["loss"] - lo["loss"]
            ok = abs(const + dN + dS + dr + quad - dl) <= 1e-7 * max(abs(dl), abs(dN), abs(dS), abs(dr), quad, 1e-300)
            print(head + " | %+10.3e = %+10.2e %+10.2e %+10.2e %+10.2e %+10.2e | %+10.2e %+10.2e %6.3f%s" % (
                dl, const, dN, dS, dr, quad, eN, eS, 1.0 - dot(rest, rest) / max(dot(Dv, Dv), 1e-300),
                "" if ok else "  <- SUM CHECK FAILED"), flush=True)
    bundle["combos"][cn] = entry
    print("   (%s done in %.1f min)" % (cn, (time.time() - t0) / 60), flush=True)
    del ctx, prob, st
    L.free_memory()
out = os.path.join(os.getcwd(), "g4_ends_%d.pt" % k_w)
torch.save(bundle, out)
print("g4_ends: wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))
print("g4_ends: done")
