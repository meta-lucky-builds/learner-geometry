"""SPEC11 section 10: build spec11/bundle/ and spec11_bundle.zip (run on this machine, CPU).

Copies (bytes) part9lib.py, the three start files, the four MNIST idx files and the three spec11 scripts; writes
cfg11.json (CONFIG cell exactly as loc.py), ref11.json (start numbers computed HERE through spec11_common.load from the
bundle itself) and MANIFEST.json (md5 and size of every file). Reads $TCR_DATA/ read-only; writes only inside spec11/.
Environment: SPEC11_THREADS (optional CPU threads for the ref11 computation).
"""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import os
import sys
import json
import shutil
import zipfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import spec11_common as S         # noqa: E402

KEEP = DATA + "/keep"
BUNDLE = os.path.join(HERE, "bundle")
ZIP = os.path.join(HERE, "spec11_bundle.zip")
WORK = os.path.join(HERE, "_bundle_work")          # library RUN_DIR while computing ref11.json


def copy(src, rel):
    dst = os.path.join(BUNDLE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    assert S.file_md5(src) == S.file_md5(dst), rel


def main():
    os.makedirs(BUNDLE, exist_ok=True)
    copy(os.path.join(KEEP, "part9lib.py"), "part9lib.py")
    for c in ("tanh-init-s0", "gelu-init-s0", "linear-init-s0"):
        for ext in (".pt", ".json"):
            copy(os.path.join(KEEP, "part9_run", "starts", c + ext), os.path.join("run", "starts", c + ext))
    for fn in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        copy(os.path.join(KEEP, "mnist_data", "MNIST", "raw", fn), os.path.join("mnist", fn))
    for fn in ("spec11_common.py", "spec11_run.py", "spec11_report.py"):
        copy(os.path.join(HERE, fn), fn)
    CFG = S.config_from_notebook(os.path.join(KEEP, "evaluation_point_tests.ipynb"))
    with open(os.path.join(BUNDLE, "cfg11.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, sort_keys=True)

    # ref11.json: the only full-size computation (three Jacobians, two test passes)
    th = os.environ.get("SPEC11_THREADS")
    L, CFGb = S.load(BUNDLE, "cpu", int(th) if th else None, WORK)
    ref = {}
    for name in ("tanh-init-s0", "gelu-init-s0", "linear-init-s0"):
        ctx = L.Ctx9(S.COMBOS[name], os.path.join(BUNDLE, "run"))
        combo = S.combo_from_ctx(L, CFGb, ctx)
        ref[name] = S.env_numbers(L, combo, name != S.LIN_NAME)
        print(name, ref[name], flush=True)
        del ctx, combo
        L.free_memory()
    with open(os.path.join(BUNDLE, "ref11.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1, sort_keys=True)

    man = {}
    for root, _, files in os.walk(BUNDLE):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, BUNDLE).replace(os.sep, "/")
            if rel == "MANIFEST.json":
                continue
            man[rel] = dict(md5=S.file_md5(p), size=os.path.getsize(p))
    with open(os.path.join(BUNDLE, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    tmp = ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(man) + ["MANIFEST.json"]:
            z.write(os.path.join(BUNDLE, rel), rel)
    os.replace(tmp, ZIP)
    print(f"wrote {ZIP} ({os.path.getsize(ZIP) / 1e6:.1f} MB, {len(man) + 1} files)", flush=True)


if __name__ == "__main__":
    main()
