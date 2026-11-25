"""Local test bed for spec10_run.py: builds a fresh MODE-tiny run directory with the library's own runner
(part9lib.build_or_load_start and part9lib.run_combo_cases9), containing
  the starts of every G4 combination, of the Part A combinations and of the linear control,
  the Part A cases of AJ(1) for direction 'true', size s2, both signs (start = init combinations),
  ALL G4 cases (every learner of G4_LEARNERS, every tau).
CPU, synthetic data (no download).

usage:  PART9_NB=<notebook> PART9LIB_DIR=<folder of part9lib.py> PART9_NO_DOWNLOAD=1 python make_tiny10.py <run_dir>
        (MODE is forced to "tiny"; the run directory must not exist yet or be one made by this script)
"""
import os, sys, json, re

rd = os.path.abspath(sys.argv[1])
nbp = os.path.abspath(os.environ["PART9_NB"])
os.chdir(os.path.dirname(nbp))
sys.path.insert(0, os.environ.get("PART9LIB_DIR", os.getcwd()))
nb = json.load(open(nbp, encoding="utf-8"))
src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "S1_FRAC, S2_FRAC" in "".join(c["source"])][0]
src, k = re.subn(r'MODE = "\w+"', 'MODE = "tiny"', src, count=1)
assert k == 1
g = {}
exec(src, g)
CFG = g["CFG"]
CFG["MNIST_DOWNLOAD"] = False
CFG["RUN_DIR"] = rd

import part9lib as L
L.configure(CFG)
L.set_device("cpu")
L._load_mnist_raw()
os.makedirs(rd, exist_ok=True)
sig = L.config_signature()
print("make_tiny10: MODE %s data_source %s run_dir %s" % (CFG["MODE"], L.DATA_SOURCE, rd), flush=True)

L.build_or_load_start(L.linear_combo(), rd, verbose=True)
for cb in L.combos("A"):
    if cb["start"] != "init":
        continue
    L.build_or_load_start(cb, rd, verbose=True)
    cases = [c for c in L.plan_A(cb) if c["learner"] == "AJ(1)" and c["dir"] == "true" and c["size"] == "s2"]
    print(L.cname9(cb), "Part A", [c["name"] for c in cases], L.run_combo_cases9(cb, cases, rd, sig), flush=True)
for cb in L.combos("G4"):
    L.build_or_load_start(cb, rd, verbose=True)
    cases = L.plan_G4(cb)
    print(L.cname9(cb), "G4", len(cases), "cases", L.run_combo_cases9(cb, cases, rd, sig), flush=True)
failed = os.listdir(os.path.join(rd, "failed")) if os.path.isdir(os.path.join(rd, "failed")) else []
print("make_tiny10: done; failed cases: %s" % (failed or "none"), flush=True)
sys.exit(1 if failed else 0)
