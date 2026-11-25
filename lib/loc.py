"""Local loader: the rig's library + config in MODE full, real MNIST read from the downloaded idx files (no torchvision here),
start weights from the downloaded run directory. CPU."""
import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import os, sys, json, struct, numpy as np, torch
KEEP = DATA + "/keep"
RD = KEEP + "/part9_run"

def _idx(path):
    b = open(path, "rb").read()
    magic, n = struct.unpack(">II", b[:8])
    if magic == 2051:
        r, c = struct.unpack(">II", b[8:16]); return torch.from_numpy(np.frombuffer(b, np.uint8, offset=16).reshape(n, r, c).copy())
    assert magic == 2049
    return torch.from_numpy(np.frombuffer(b, np.uint8, offset=8).astype(np.int64))

def load(threads=6):
    os.chdir(KEEP); sys.path.insert(0, KEEP)
    nb = json.load(open(KEEP + "/evaluation_point_tests.ipynb", encoding="utf-8"))
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "S1_FRAC, S2_FRAC" in "".join(c["source"])][0]
    g = {}; exec(src, g); CFG = g["CFG"]; assert CFG["MODE"] == "full"
    CFG["MNIST_DOWNLOAD"] = False; CFG["RUN_DIR"] = OUT + "/scratch_run"
    os.makedirs(CFG["RUN_DIR"], exist_ok=True)
    torch.set_num_threads(threads)
    import part9lib as L
    L.configure(CFG); L.set_device("cpu")
    raw = KEEP + "/mnist_data/MNIST/raw/"
    L._MNIST_RAW = (_idx(raw + "train-images-idx3-ubyte"), _idx(raw + "train-labels-idx1-ubyte"))
    L._MNIST_TEST_RAW = (_idx(raw + "t10k-images-idx3-ubyte"), _idx(raw + "t10k-labels-idx1-ubyte"))
    L.DATA_SOURCE = "mnist"
    return L, CFG
