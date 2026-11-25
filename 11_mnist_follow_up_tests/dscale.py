import os; DATA = os.environ.get("TCR_DATA", "data"); OUT = os.environ.get("TCR_OUT", "outputs")  # data / output folders, see README
import torch, glob, math
for f in sorted(glob.glob(DATA + "/keep/spec10_out/bundle_*.pt")):
    B = torch.load(f, map_location="cpu", weights_only=True)
    for cn, e in B["combos"].items():
        row = []
        for tau in (0.1, 0.25, 0.5, 1):
            a, b = e["ends"].get("large|ZERO|tau%g|DJ(0.25)" % tau), e["ends"].get("large|ZERO|tau%g|DJ(1)" % tau)
            row.append(float((a.double() - b.double()).pow(2).mean().sqrt()) if a is not None and b is not None else float("nan"))
        ex = [math.log(row[i + 1] / row[i]) / math.log(t2 / t1) for i, (t1, t2) in enumerate(((0.1, 0.25), (0.25, 0.5), (0.5, 1)))]
        print("%-19s |D|rms at tau .1 .25 .5 1: %s | exponents %s" % (cn, " ".join("%.3e" % x for x in row), " ".join("%.2f" % x for x in ex)))
