import json, sys, os, math
import numpy as np
from collections import Counter, defaultdict
base = sys.argv[1]
R = json.load(open(os.path.join(base, "results_part8.json"), encoding="utf-8"))
print("mode", R["mode"], "| data_source", R["data_source"], "| signature", R["signature"], "| written", R["written"], "| torch", R["environment"].get("torch"), "| cases", len(R["cases"]), "| failed", len(R["failed"]), "| missing", len(R["missing"]))
print("missing by combination:", dict(Counter(m.split("__")[0] for m in R["missing"])))
H = R["H_table"]


def fl(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


print("\n== PRIMARY extrapolated rules: exact even-part values (from the notebook's Analysis) ==")
allx = []
for h in H:
    if h["rule"] in ("H1x", "H2x", "H3x", "H4x") and h["used"]:
        for c in h["cases"]:
            c = dict(c); c["rule"] = h["rule"]; c["act"], c["start"] = h["group"]; allx.append(c)
print("units:", len(allx), " matched:", sum(bool(c["match"]) for c in allx), " excluded lists:", sum(len(h.get("excluded", [])) for h in H if h["rule"] in ("H1x", "H2x", "H3x", "H4x")))
for act in ("tanh", "gelu"):
    print("  ", act)
    for ln in sorted({c["learner"] for c in allx}, key=lambda s: (s.split("(")[0], float(s.split("(")[1].split(",")[0].rstrip(")")))):
        xs = [c for c in allx if c["act"] == act and c["learner"] == ln]
        e = [abs(fl(c["rho_x"]) - fl(c["f"])) / max(1.0, abs(fl(c["f"]))) for c in xs]
        e1 = [abs(fl(c["rho_s1"]) - fl(c["f"])) / max(1.0, abs(fl(c["f"]))) for c in xs]
        cs = [abs(fl(c["cos_x"])) for c in xs if abs(fl(c["f"])) >= 0.05]
        w = xs[int(np.argmax(e))]
        print("     %-15s n=%3d  rel err of limit: median %.1e max %.1e | at size s1: median %.1e max %.1e | min |cos| %s | worst: %s %s %s rho_x %+.5f f %+.5f" % (
            ln, len(xs), np.median(e), max(e), np.median(e1), max(e1), ("%.6f" % min(cs)) if cs else "   -    ", w["combo"], w["dir"], "", fl(w["rho_x"]), fl(w["f"])))
tol = {"H1x": 0.02, "H2x": 0.02, "H3x": 0.03, "H4x": 0.05}
marg = [abs(fl(c["rho_x"]) - fl(c["f"])) / max(1.0, abs(fl(c["f"]))) / tol[c["rule"]] for c in allx]
print("   error as a fraction of the allowed tolerance: median %.4f, 99th pct %.3f, max %.3f" % (np.median(marg), np.percentile(marg, 99), max(marg)))
print("   by direction (max rel err of limit): " + ", ".join("%s %.1e" % (d, max(abs(fl(c["rho_x"]) - fl(c["f"])) / max(1.0, abs(fl(c["f"]))) for c in allx if c["dir"] == d)) for d in ("eigA", "eigB", "eigC", "eigD", "label", "rand")))
print("   by start (max): " + ", ".join("%s/%s %.1e" % (a, s, max(abs(fl(c["rho_x"]) - fl(c["f"])) / max(1.0, abs(fl(c["f"]))) for c in allx if c["act"] == a and c["start"] == s)) for a in ("tanh", "gelu") for s in ("init", "trained")))
print("\n== convergence conditions of H3x/H4x; H7x; H0; H6 ==")
for h in H:
    if h["rule"] in ("H7x", "H0", "H6", "H5b") and h["used"]:
        if h["rule"] == "H7x":
            print("   H7x %s/%s: %d/%d  max nr(s2) %.5f" % (h["group"][0], h["group"][1], h["matched"], h["used"], max(fl(c["nr_s2"]) for c in h["cases"])))
        elif h["rule"] == "H0":
            print("   H0: %d/%d  max %.2e / %.2e" % (h["matched"], h["used"], max(fl(c["vL_over_dL"]) for c in h["cases"]), max(fl(c["vNP_over_dNP"]) for c in h["cases"])))
        elif h["rule"] == "H6":
            print("   H6 %s/%s: %d/%d  rho [%.7f, %.7f] min cos %.9f  excluded %d" % (h["group"][0], h["group"][1], h["matched"], h["used"], min(fl(c["rho"]) for c in h["cases"]), max(fl(c["rho"]) for c in h["cases"]), min(fl(c["cos"]) for c in h["cases"]), len(h.get("excluded", []))))
        else:
            print("   H5b %s/%s: %d/%d  max %.4f" % (h["group"][0], h["group"][1], h["matched"], h["used"], max(fl(c["Clin_rel_err"]) for c in h["cases"])))
print("\n== H5 full path law ==")
for h in H:
    if h["rule"] == "H5" and h["used"]:
        cs = h["cases"]
        print("   %s/%s: %d/%d  cos [%.6f, %.6f]  ratio [%.5f, %.5f]  kept SV %d..%d" % (h["group"][0], h["group"][1], h["matched"], h["used"], min(fl(c["law_cos"]) for c in cs), max(fl(c["law_cos"]) for c in cs), min(fl(c["law_ratio"]) for c in cs), max(fl(c["law_ratio"]) for c in cs), min(c["kept"] for c in cs), max(c["kept"] for c in cs)))
        for pre in ("GD(0.25)", "GD(1.5)", "SGD"):
            ys = [c for c in cs if c["learner"].startswith(pre)]
            print("        %-9s law cos min %.6f  ratio median %.4f  | measured even rho %.3f..%.3f  cos %.3f..%.3f" % (pre, min(fl(c["law_cos"]) for c in ys), np.median([fl(c["law_ratio"]) for c in ys]), min(fl(c["even_rho"]) for c in ys), max(fl(c["even_rho"]) for c in ys), min(fl(c["even_cos"]) for c in ys), max(fl(c["even_cos"]) for c in ys)))
print("\n== literal single-size rules: every failing case ==")
for h in H:
    if h.get("label", "").startswith("literal") and h["used"] and h["rule"] != "H7":
        bad = [c for c in h["cases"] if not c["match"]]
        print("   %s %s/%s: %d/%d  failing by (learner, dir, size): %s" % (h["rule"], h["group"][0], h["group"][1], h["matched"], h["used"], dict(Counter((c["learner"], c["dir"], c["size"]) for c in bad))))
print("\n== order check (official R4): flags ==")
for g, T in R["R_tables"].items():
    if g == "R6" or "R4_order_check" not in T:
        continue
    rows = [r for r in T["R4_order_check"] if "ratio" in r]
    if not rows:
        continue
    big = [r for r in rows if fl(r["err_s1"]) > 1e-4]
    rt = [fl(r["ratio"]) for r in big]
    fg = [r for r in rows if r.get("flag")]
    print("   %-13s rows %d (with error > 1e-4: %d)  ratio median %.3f  5%%-95%% [%.2f, %.2f]  flagged %d %s" % (g, len(rows), len(big), np.median(rt), np.percentile(rt, 5), np.percentile(rt, 95), len(fg), [(r["learner"], r["dir"], round(fl(r["ratio"]), 2), "%.1e" % fl(r["err_s1"])) for r in fg][:6]))
print("\n== relu: what the official analysis says ==")
for h in H:
    if h["group"][0] == "relu" and (h["used"] or h.get("excluded")):
        ex = h.get("excluded", [])
        print("   %s relu/%s: used %d, excluded %d  reasons %s" % (h["rule"], h["group"][1], h["used"], len(ex), dict(Counter(str(e[1]).split(":")[-1].strip() for e in ex))))
