"""Analysis-only pass for alignment_fraction_test.ipynb: evaluates the pre-registered rules on the cases that are
already on disk (part8_run/done + part8_run/vecs), with the notebook's own Analysis class and the exact vectors.
Runs NO learner and changes no case file. Use after stopping an unfinished grid run.
Run from the folder that holds the notebook, part8lib.py and part8_run/."""
import json, os, sys, glob, io, contextlib
sys.path.insert(0, os.getcwd())
nb = json.load(open("alignment_fraction_test.ipynb", encoding="utf-8"))
cfg_src = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code" and "CONFIG (every constant" in "".join(c["source"])][0]
mode = os.environ.get("PART8_MODE")
if mode:                                            # local testing only
    import re
    cfg_src = re.sub(r'^MODE = "[a-z]+"', 'MODE = "%s"' % mode, cfg_src, count=1, flags=re.M)
ns = {}
exec(cfg_src, ns)
CFG = ns["CFG"]
import part8lib as L
L.configure(CFG)
L.set_device("cpu")                                 # analysis only: vectors are loaded on the CPU
L._load_mnist_raw()
RUN_DIR = CFG["RUN_DIR"]
sig = L.config_signature()
sigs = {}
for p in glob.glob(os.path.join(RUN_DIR, "done", "*.json"))[:50]:
    s = json.load(open(p, encoding="utf-8")).get("signature")
    sigs[s] = sigs.get(s, 0) + 1
print("MODE=%s data_source=%s signature=%s | signatures found in done/ (sample): %s" % (CFG["MODE"], L.DATA_SOURCE, sig, sigs))
if sig not in sigs:
    raise SystemExit("STOP: configuration signature does not match the case files; nothing analysed.")
START_INFOS = {}
for p in sorted(glob.glob(os.path.join(RUN_DIR, "starts", "*.json"))):
    d = L.read_json(p)
    START_INFOS[d["combo"]] = d
ILL = [k for k, v in START_INFOS.items() if v.get("ill_conditioned")]
lin = L.linear_combo()
PLAN = [dict(combo=lin, cases=L.plan_combo(lin), kernel_only=True)]
PLAN += [dict(combo=cb, cases=L.plan_combo(cb)) for cb in L.all_combos() if L.combo_name(cb) not in ILL]
n_plan = sum(len(it["cases"]) for it in PLAN)
A = L.Analysis(PLAN, RUN_DIR, START_INFOS)
print("planned %d cases; on disk with this signature: %d done, %d failed, %d not run" % (n_plan, len(A.recs), len(A.failed), len(A.missing)))
by = {}
for it in PLAN:
    nm = L.combo_name(it["combo"])
    by[nm] = (sum(1 for c in it["cases"] if c["name"] in A.recs), len(it["cases"]))
print("completeness per combination: " + ", ".join("%s %d/%d" % (k, a, b) for k, (a, b) in by.items()))
print("NOTE: this is an analysis-only pass on an INTERRUPTED grid; combinations that are not complete are listed above and their")
print("      rules are evaluated on whatever finished (missing cases are listed by the analysis as excluded / missing).\n")
H, R = A.run()
tests = [dict(test="(analysis-only pass: unit tests ran in the original notebook execution, see run log)", result="-")]
L.save_outputs(H, R, A.recs, A.failed, A.missing, tests, START_INFOS, {}, A.lines)
with open("analysis_only_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(A.lines))
print("wrote analysis_only_output.txt")
