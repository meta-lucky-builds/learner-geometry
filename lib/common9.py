"""Shared loader for the stage-09/10 check scripts: executes the CONFIG cell of the stage-09 notebook with a chosen MODE
and configures the library part9lib. No downloads: MNIST idx files must already be in the configured data folder."""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "09_evaluation_point", "evaluation_point_tests.ipynb")


def load(mode="tiny", run_dir=None):
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    cfg_src = ["".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "S1_FRAC, S2_FRAC" in "".join(c["source"])][0]
    cfg_src = cfg_src.replace('MODE = "quick"', 'MODE = "%s"' % mode)
    assert ('MODE = "%s"' % mode) in cfg_src
    g = {}
    exec(cfg_src, g)
    CFG = g["CFG"]
    CFG["MNIST_DOWNLOAD"] = False
    if run_dir:
        CFG["RUN_DIR"] = run_dir
    import part9lib as L
    L.configure(CFG)
    L._load_mnist_raw()
    return L, CFG
