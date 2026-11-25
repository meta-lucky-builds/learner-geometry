import json, glob, math, sys
recs = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/records_*.json"))]
combos = {}
for r in recs: combos.update(r["combos"])
RES = recs[0]["cfg"]["G4_RES_MAX"]
BAD = ("DIVERGED", "AJ-FAILED", "AJ-INCONSISTENT", "ERROR")
us = lambda e: e.get("learner_status") not in BAD and isinstance(e.get("train_res"), float) and math.isfinite(e["train_res"]) and e["train_res"] <= RES
names = sorted(combos, key=lambda c: (-combos[c]["n"], c))
S1 = S2 = both = n = 0; S3 = n3 = 0
print("%-19s %5s | %9s %9s | %8s %8s %8s %8s %8s | %7s %7s | %8s %8s | %s" % ("combo", "tau", "dlossZ", "rel", "const", "dN", "dS", "drest", "quad", "eN", "eS", "dloss*", "dacc", "S1 S2  gN  gS"))
rows = []
for cn in names:
    for b in combos[cn]["blocks"]:
        Z = {e["learner"]: e for e in b["ZERO"]["ends"] if us(e)}
        sp = b["ZERO"]["splits"].get("DJ(0.25)")
        if "DJ(1)" not in Z or "DJ(0.25)" not in Z or sp is None: continue
        dl = Z["DJ(0.25)"]["loss"] - Z["DJ(1)"]["loss"]
        dls = Z["DJ(0.25)"]["loss_star"] - Z["DJ(1)"]["loss_star"]
        dacc = Z["DJ(0.25)"]["acc"] - Z["DJ(1)"]["acc"]
        s1, s2 = dls < 0, sp["drest"] < 0
        n += 1; S1 += s1; S2 += s2; both += (s1 and s2)
        if b["tau"] <= 0.5: n3 += 1; S3 += (s1 and s2)
        rows.append((cn, b["tau"], dl, sp, dls, dacc, Z))
        print("%-19s %5g | %+9.2e %+9.2e | %+8.1e %+8.1e %+8.1e %+8.1e %+8.1e | %+7.3f %+7.3f | %+8.1e %+8.4f | %d  %d  %+.1e %+.1e" % (
            cn, b["tau"], dl, dl / Z["DJ(1)"]["loss"], sp["const"], sp["dN"], sp["dS"], sp["drest"], sp["quad"], sp["eN"], sp["eS"], dls, dacc, s1, s2, sp["gN"], sp["gS"]))
print("\nS1 (loss* better) %d/%d   S2 (drest<0) %d/%d   both %d/%d   S3 (both, tau<=0.5) %d/%d  [rules: S1>=24, S2>=24, S3>=18 of 24]" % (S1, n, S2, n, both, n, S3, n3))
print("\nall usable ZERO learners per block: loss minus loss of DJ(1), units 1e-6 (f in brackets); accuracy in %")
for cn, tau, dl, sp, dls, dacc, Z in rows:
    o = sorted(Z.values(), key=lambda e: e["f_nominal"])
    print("%-19s %5g  " % (cn, tau) + "  ".join("%s[%+.2f] %+.0f" % (e["learner"], e["f_nominal"], 1e6 * (e["loss"] - Z["DJ(1)"]["loss"])) for o_ in [o] for e in o_)
          + "   | acc " + " ".join("%.2f" % (100 * e["acc"]) for e in o))
