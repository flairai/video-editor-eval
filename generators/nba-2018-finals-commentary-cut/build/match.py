"""Match each shot of the segment to the indexed source footage.
Usage: match.py SEGMENT_FRAMES.npz INDEX_DIR OUT.json"""
import sys, json, glob, os, numpy as np
seg = np.load(sys.argv[1]); idx_dir, out = sys.argv[2], sys.argv[3]
st, sf, shots, fps = seg["times"], np.nan_to_num(seg["feats"].astype(np.float32)), seg["shots"], float(seg["fps"])
srcs = []
for p in sorted(glob.glob(os.path.join(idx_dir, "*.npz"))):
    z = np.load(p); srcs.append((os.path.basename(p)[:-4], z["times"], z["feats"].astype(np.float32)))
names = [s[0] for s in srcs]
F = np.nan_to_num(np.concatenate([s[2] for s in srcs])); TT = np.concatenate([s[1] for s in srcs])
FID = np.concatenate([np.full(len(s[1]), i) for i, s in enumerate(srcs)])
rng = np.random.default_rng(0)
results = []
for k, (a, b) in enumerate(shots):
    frames = list(range(a, b + 1, 1 if b - a < 15 else 3)) or [a]
    S = sf[frames] @ F.T                                      # cosine similarity
    pairs = []
    for r, fi in enumerate(frames):
        top = np.argpartition(-S[r], 5)[:5]
        for j in top:
            if S[r, j] >= 0.80:
                pairs.append((st[fi], FID[j], TT[j], S[r, j]))
    best = None
    if pairs:
        P = np.array(pairs, dtype=np.float64)
        hyps = []
        for i in range(len(P)):                                # rate 1 and freeze hypotheses from single pairs
            hyps.append((P[i, 1], 1.0, P[i, 2] - P[i, 0]))
            hyps.append((P[i, 1], 0.0, P[i, 2]))
        for _ in range(300):                                   # general rate from two pairs
            i, j = rng.integers(0, len(P), 2)
            if P[i, 1] != P[j, 1] or abs(P[i, 0] - P[j, 0]) < 0.2: continue
            rate = (P[j, 2] - P[i, 2]) / (P[j, 0] - P[i, 0])
            if 0.15 <= rate <= 1.6: hyps.append((P[i, 1], rate, P[i, 2] - rate * P[i, 0]))
        for fid, rate, off in hyps:
            m = P[:, 1] == fid
            res = np.abs(P[m, 2] - (rate * P[m, 0] + off))
            inl = res < 0.12
            seg_times_hit = len(set(np.round(P[m][inl, 0], 3)))
            score = seg_times_hit + P[m][inl, 3].sum() * 0.01
            if best is None or score > best[0]:
                best = (score, int(fid), float(rate), float(off), seg_times_hit, float(P[m][inl, 3].mean()) if inl.any() else 0.0)
    rec = {"shot": k, "seg_start": round(float(st[a]), 3), "seg_end": round(float(st[b]) + 1 / fps, 3), "n_sampled": len(frames)}
    if best and best[4] >= max(2, 0.5 * len(frames)) and best[5] >= (0.86 if len(frames) >= 6 else 0.9):
        _, fid, rate, off, hit, msim = best
        rec.update({"source": names[fid], "rate": round(rate, 3), "src_start": round(rate * rec["seg_start"] + off, 3),
                    "src_end": round(rate * rec["seg_end"] + off, 3), "coverage": round(hit / len(frames), 2), "mean_sim": round(msim, 3)})
    else:
        rec.update({"source": None, "best_partial": None if not best else {"source": names[best[1]], "coverage": round(best[4] / len(frames), 2), "mean_sim": round(best[5], 3)}})
    results.append(rec)
json.dump({"sources": names, "shots": results}, open(out, "w"), indent=1)
m = [r for r in results if r["source"]]
print("matched %d/%d shots, %.1f of %.1f s" % (len(m), len(results), sum(r["seg_end"] - r["seg_start"] for r in m), results[-1]["seg_end"]))
for r in results:
    if r["source"]:
        print("  %02d %6.2f-%6.2f -> %s %8.2f-%8.2f rate %.2f cov %.2f sim %.3f" % (r["shot"], r["seg_start"], r["seg_end"], r["source"], r["src_start"], r["src_end"], r["rate"], r["coverage"], r["mean_sim"]))
    else:
        bp = r["best_partial"]
        print("  %02d %6.2f-%6.2f -> no match%s" % (r["shot"], r["seg_start"], r["seg_end"], "" if not bp else "  (best %s cov %.2f sim %.3f)" % (bp["source"], bp["coverage"], bp["mean_sim"])))
