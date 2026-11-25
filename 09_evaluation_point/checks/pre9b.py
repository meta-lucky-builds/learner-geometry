import sys
sys.argv = ["pre9.py", "none"]
exec(open("pre9.py", encoding="utf-8").read())
prob, th0 = L.toy_problem(); st = L.Start(prob, th0); d = torch.ones(1); s = 0.02
print("solvable model u^2 - v^2, s = 0.02, even part: value [cos] prediction")
for nm, fn, pred in [("SLa(2,1)", lambda t: run_SLa(prob, st, t, 2, 1.0), .75), ("SLa(8,2)", lambda t: run_SLa(prob, st, t, 8, 2.0), .6875), ("SLa(4,-1)", lambda t: run_SLa(prob, st, t, 4, -1.0), .125),
                     ("SLa(2,0.5)", lambda t: run_SLa(prob, st, t, 2, 0.5), .5),
                     ("TJ(0.5,0.5)", lambda t: run_TJ(prob, st, t, .5, .5), .5), ("TJ(1.5,0.5)", lambda t: run_TJ(prob, st, t, 1.5, .5), .5), ("TJ(0.5,1)", lambda t: run_TJ(prob, st, t, .5, 1.), .6),
                     ("TJ(1.5,1)", lambda t: run_TJ(prob, st, t, 1.5, 1.), 2.5 / 3.5), ("TJ(8,1)", lambda t: run_TJ(prob, st, t, 8., 1.), .9), ("TJ(3,0.25)", lambda t: run_TJ(prob, st, t, 3., .25), -1.0),
                     ("TM(0.5,0.5)", lambda t: run_TM(prob, st, t, s, .5, .5), .5), ("TM(1.5,0.5)", lambda t: run_TM(prob, st, t, s, 1.5, .5), .5), ("TM(0.5,1)", lambda t: run_TM(prob, st, t, s, .5, 1.), .6),
                     ("TM(1.5,1)", lambda t: run_TM(prob, st, t, s, 1.5, 1.), 2.5 / 3.5), ("TM(6,1)", lambda t: run_TM(prob, st, t, s, 6., 1.), .875), ("TM(1.5,0)", lambda t: run_TM(prob, st, t, s, 1.5, 0.), -1.0)]:
    r, c, ok = rho_even(prob, st, d, s, fn)
    print("   %-12s %+.4f [%+.3f]  pred %+.4f %s" % (nm, r, c, pred, "" if ok else "NOT-OK"), flush=True)
