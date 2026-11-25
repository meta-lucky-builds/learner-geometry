import json, sys, collections
rows = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    rows[(r["combo"], r["tau"], r["kind"], r["learner"])] = r   # later rows overwrite repeats
print("unique rows", len(rows), "statuses", collections.Counter(r["status"] for r in rows.values()))
blocks = sorted({k[:3] for k in rows})
cnt = collections.Counter(); tot = collections.Counter()
print("%-14s %-4s %-5s | acc2: f-1   f0    f.375 f.43 | mse2 x1e5 rel to f0: f-1  f.375 f.43 | cka2: f-1   f0    f.375  f.43 | acc1 f-1 f0 f.43 | cka1 f-1 f0 f.43" % ("start","tau","kind"))
for b in blocks:
    g = lambda L: rows.get(b + (L,))
    m1, z, s4, d25 = g("DJ(1.5)"), g("DJ(1)"), g("SL(4)"), g("DJ(0.25)")
    if not all([m1, z, s4, d25]): print(b, "incomplete"); continue
    print("%-14s %-4s %-5s | %.4f %.4f %.4f %.4f | %+7.1f %+7.1f %+7.1f | %.4f %.4f %.4f %.4f | %.4f %.4f %.4f | %.4f %.4f %.4f" % (b[0], b[1], b[2],
        m1["acc2"], z["acc2"], s4["acc2"], d25["acc2"],
        (m1["mse2"]-z["mse2"])*1e5, (s4["mse2"]-z["mse2"])*1e5, (d25["mse2"]-z["mse2"])*1e5,
        m1["cka2"], z["cka2"], s4["cka2"], d25["cka2"], m1["acc1"], z["acc1"], d25["acc1"], m1["cka1"], z["cka1"], d25["cka1"]))
    k = b[2]; tot[k] += 1
    cnt[k, "acc2 f.43>f0"] += d25["acc2"] > z["acc2"]
    cnt[k, "acc2 f.43==f0"] += d25["acc2"] == z["acc2"]
    cnt[k, "acc2 f-1<f0"] += m1["acc2"] < z["acc2"]
    cnt[k, "acc2 f-1==f0"] += m1["acc2"] == z["acc2"]
    cnt[k, "mse2 f.43<f0"] += d25["mse2"] < z["mse2"]
    cnt[k, "mse2 f-1<f0"] += m1["mse2"] < z["mse2"]
    cnt[k, "mse2 monotone rising in f"] += m1["mse2"] < z["mse2"] < s4["mse2"] <= d25["mse2"] + 1e-12
    cnt[k, "cka2 f.43>f0"] += d25["cka2"] > z["cka2"]
    cnt[k, "cka2 f-1<f0"] += m1["cka2"] < z["cka2"]
    cnt[k, "acc1 f.43>f0"] += d25["acc1"] > z["acc1"]
    cnt[k, "acc1 f-1<f0"] += m1["acc1"] < z["acc1"]
    cnt[k, "mse1 f.43<f0"] += d25["mse1"] < z["mse1"]
    cnt[k, "mse1 f-1<f0"] += m1["mse1"] < z["mse1"]
    cnt[k, "cka1 f.43>f0"] += d25["cka1"] > z["cka1"]
    cnt[k, "cka1 f-1<f0"] += m1["cka1"] < z["cka1"]
print()
for (k, name), v in sorted(cnt.items()):
    print("%-5s %-28s %d / %d" % (k, name, v, tot[k]))
