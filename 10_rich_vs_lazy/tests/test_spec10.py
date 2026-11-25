"""Local end-to-end test of spec10_run.py and spec10_report.py (SPEC10 sec. 10 item 3). MODE tiny, CPU, synthetic data.

Steps (every program runs as a subprocess, as on the rig):
  1. make_tiny10.py builds a fresh run directory review10/tiny_run10 with the library's own runner;
  2. the worker (1 of 1) and the report run on it in review10/runs/main/ (a copy of the notebook is placed there so that
     spec10_out/ lands next to it, inside review10/);
  3. the same for the three negative controls SPEC10_BREAK=target, kappa, repro (review10/runs/break_*/).
Asserts: exit codes 0; V1 and V2 <= 1e-12 (CPU), V3-V7 pass; every planned end point present (planned list rebuilt here from
CFG and the stored G4 records); the report runs and writes its files; each negative control makes its check FAIL
(target -> V3 explicit ZERO check, kappa -> V7, repro -> V1) and the report says INVALID.
It deletes only what it made itself (review10/tiny_run10 and review10/runs) before starting.

usage:  python review10/test_spec10.py          (exit code 0 = every assertion passed)
"""
import os, sys, json, re, shutil, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB = os.path.join(ROOT, "evaluation_point_tests.ipynb")
LIB = os.path.join(ROOT, "review9", "v3")
RUN = os.path.join(HERE, "tiny_run10")
RUNS = os.path.join(HERE, "runs")
TOL_CPU = 1e-12                                       # sec. 10.3: V1, V2 on CPU
PY = sys.executable

ENV = dict(os.environ, PART9LIB_DIR=LIB, PART9_MODE="tiny", PART9_NO_DOWNLOAD="1", PART9_DEVICE="cpu", PART9_RUN_DIR=RUN,
           PYTHONIOENCODING="utf-8")
ENV.pop("SPEC10_BREAK", None)
ENV.pop("SPEC10_FORCE", None)

RESULTS = []


def check(what, ok):
    RESULTS.append((what, bool(ok)))
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what), flush=True)


def run(cmd, env, log):
    t0 = time.time()
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(p.stdout)
    print("  ran %s (exit %d, %.0f s, log %s)" % (" ".join(os.path.basename(c) for c in cmd[1:]), p.returncode,
                                                 time.time() - t0, os.path.relpath(log, ROOT)), flush=True)
    return p.returncode, p.stdout


for d in (RUN, RUNS):
    if os.path.exists(d):
        shutil.rmtree(d)
os.makedirs(RUNS)

print("=== 1. fresh tiny run directory with the library's own runner", flush=True)
env1 = dict(ENV, PART9_NB=NB)
rc, txt = run([PY, os.path.join(HERE, "make_tiny10.py"), RUN], env1, os.path.join(RUNS, "make_tiny10.log"))
print(txt, flush=True)
check("make_tiny10.py exit code 0", rc == 0)

# planned end points, rebuilt here from CFG and the stored G4 records (independent of the worker's own list)
os.chdir(ROOT)
sys.path.insert(0, LIB)
nb = json.load(open(NB, encoding="utf-8"))
src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "S1_FRAC, S2_FRAC" in "".join(c["source"])][0]
src = re.subn(r'MODE = "\w+"', 'MODE = "tiny"', src, count=1)[0]
g = {}
exec(src, g)
CFG = g["CFG"]
CFG["MNIST_DOWNLOAD"] = False
CFG["RUN_DIR"] = RUN
import torch
import part9lib as L
L.configure(CFG)
L._load_mnist_raw()
IMPLICIT = [("TJ", 1.5, 1.0), ("SLa", 4, 2.0), ("AJ", 1.0)]
PLANNED = {}
for cb in L.combos("G4"):
    cn = L.cname9(cb)
    for tau in CFG["G4_TAUS"]:
        zero = ["DJ(1.5)", "DJ(1)", "SL(4)", "DJ(0.25)"]
        for x, fnom in CFG["G4_LEARNERS"]:
            if any(x[0] == sp[0] and len(x) == len(sp) and all(float(a) == float(b) for a, b in zip(x[1:], sp[1:])) for sp in IMPLICIT):
                Ld = L.L_from_spec(x)
                pth = os.path.join(RUN, "done", L.safe_name(L._case9(cn, "G4", "G4", Ld, "true", f"tau{tau:g}", tau, 1)["name"]) + ".json")
                r = L.read_json(pth) if os.path.exists(pth) else None
                if r and r.get("learner_status") not in ("DIVERGED", "AJ-FAILED", "AJ-INCONSISTENT") and \
                        L._fin(r.get("train_res")) and r["train_res"] <= CFG["G4_RES_MAX"]:
                    zero.append(L.learner_name9(Ld))
        PLANNED[(cn, tau)] = {"PLAIN": ["DJ(1)", "DJ(1.5)"], "ZERO": zero}
print("planned large-block end points (from the stored records): %s" % {"%s tau=%g" % k: v for k, v in PLANNED.items()}, flush=True)


def one(name, brk):
    d = os.path.join(RUNS, name)
    os.makedirs(d)
    nbc = os.path.join(d, os.path.basename(NB))
    shutil.copyfile(NB, nbc)
    env = dict(ENV, PART9_NB=nbc)
    if brk:
        env["SPEC10_BREAK"] = brk
    rcw, wtxt = run([PY, os.path.join(ROOT, "spec10_run.py"), "0", "1"], env, os.path.join(d, "worker.log"))
    out = os.path.join(d, "spec10_out")
    rcr, rtxt = run([PY, os.path.join(ROOT, "spec10_report.py"), out], env, os.path.join(d, "report.log"))
    res = json.load(open(os.path.join(out, "spec10_results.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(out, "spec10_results.json")) else None
    return rcw, rcr, wtxt, rtxt, res, out


print("\n=== 2. main run (worker 0 of 1, then the report)", flush=True)
rcw, rcr, wtxt, rtxt, res, out = one("main", "")
print("\n----- full report output of the main run -----", flush=True)
print(rtxt, flush=True)
print("----- end of report output -----\n", flush=True)
check("worker exit code 0", rcw == 0)
check("report exit code 0", rcr == 0)
check("report wrote spec10_report.txt and spec10_results.json",
      res is not None and os.path.exists(os.path.join(out, "spec10_report.txt")))
check("records_0.json and bundle_0.pt written into spec10_out/",
      os.path.exists(os.path.join(out, "records_0.json")) and os.path.exists(os.path.join(out, "bundle_0.pt")))
if res:
    V = res["validations"]
    check("V1 PLAIN reproduction <= %g (max %.3e over %d end points)" % (TOL_CPU, V["V1"]["max"], V["V1"]["n"]),
          V["V1"]["passed"] and V["V1"]["max"] <= TOL_CPU)
    check("V2 Part A reproduction <= %g (max %.3e over %d comparisons)" % (TOL_CPU, V["V2"]["max"], V["V2"]["n"]),
          V["V2"]["passed"] and V["V2"]["n"] > 0 and V["V2"]["max"] <= TOL_CPU)
    for k in ("V3", "V4", "V5", "V6", "V7"):
        check("%s passes" % k, V[k]["passed"])
    check("report says VALID (no INVALID banner)", not res["invalid"] and "INVALID" not in rtxt)
    # every planned end point present
    rec = json.load(open(os.path.join(out, "records_0.json"), encoding="utf-8"))
    bun = torch.load(os.path.join(out, "bundle_0.pt"))
    miss = []
    for (cn, tau), kinds in PLANNED.items():
        cr = rec["combos"].get(cn)
        if cr is None:
            miss.append(cn)
            continue
        b = [b for b in cr["blocks"] if b["tau"] == tau]
        for kind, lns in kinds.items():
            have = {e["learner"]: e for e in b[0][kind]["ends"]} if b else {}
            for ln in lns:
                if ln not in have or "loss" not in have[ln] or "large|%s|tau%g|%s" % (kind, tau, ln) not in bun["combos"][cn]["ends"]:
                    miss.append("%s tau=%g %s %s" % (cn, tau, kind, ln))
            extra = sorted(set(have) - set(lns))
            if extra:
                miss.append("%s tau=%g %s unexpected %s" % (cn, tau, kind, extra))
        for kind in ("PLAIN", "ZERO"):
            for key in ("+1", "-1"):
                e = cr["small"][kind]["signs"].get(key, {})
                if "loss_NP" not in e or "loss_TH" not in e or any(
                        "small|%s|%s|%s" % (kind, key, x) not in bun["combos"][cn]["ends"] for x in ("L", "NP", "TH")):
                    miss.append("%s small %s %s" % (cn, kind, key))
    lin = rec.get("linear") or {}
    if sorted(e["learner"] for e in lin.get("ends", []) if "loss" in e) != sorted(["DJ(1.5)", "DJ(1)", "DJ(0.25)"]):
        miss.append("linear control end points")
    check("every planned end point present (records and bundle)%s" % (": missing %s" % miss if miss else ""), not miss)
    statuses = sorted({e["learner_status"] for cr in rec["combos"].values() for b in cr["blocks"] for k in ("PLAIN", "ZERO")
                       for e in b[k]["ends"]})
    print("  learner statuses seen in the large block: %s" % statuses, flush=True)

print("\n=== 3. negative controls (each must make its check FAIL and the report say INVALID)", flush=True)
for name, brk, vk, sub in (("break_target", "target", "V3", "zero_check_passed"), ("break_kappa", "kappa", "V7", "passed"),
                           ("break_repro", "repro", "V1", "passed")):
    rcw, rcr, wtxt, rtxt, res, out = one(name, brk)
    check("SPEC10_BREAK=%s: worker and report exit 0" % brk, rcw == 0 and rcr == 0)
    check("SPEC10_BREAK=%s: loud banner printed by the worker and break written into the records" % brk,
          "SPEC10_BREAK=%s IS SET" % brk in wtxt and
          json.load(open(os.path.join(out, "records_0.json"), encoding="utf-8"))["BREAK"] == brk)
    if res is None:
        check("SPEC10_BREAK=%s: report results exist" % brk, False)
        continue
    V = res["validations"]
    lines = [l for l in rtxt.splitlines() if l.strip().startswith(vk + " ") or "INVALID" in l]
    for l in lines:
        print("      | " + l, flush=True)
    check("SPEC10_BREAK=%s: %s %s FAILS" % (brk, vk, "explicit ZERO check" if sub != "passed" else ""),
          V[vk][sub] is False and V[vk]["passed"] is False)
    check("SPEC10_BREAK=%s: report says INVALID" % brk, res["invalid"] and "INVALID" in rtxt)
    others = [k for k in sorted(V) if k != vk and not V[k]["passed"]]
    print("      other validations failing under this break: %s" % (others or "none"), flush=True)

nf = sum(1 for _, ok in RESULTS if not ok)
print("\n=== test_spec10: %d checks, %d failed -> %s" % (len(RESULTS), nf, "ALL PASS" if nf == 0 else "FAILURES"), flush=True)
sys.exit(1 if nf else 0)
