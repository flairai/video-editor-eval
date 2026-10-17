"""Refine matched shots to exact frames by decoding the source at full rate.
Usage: refine.py SEGMENT_FRAMES.npz MATCHES.json SRC_ROOT OUT.json"""
import sys, json, glob, os, av, numpy as np
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from vfeat import feat
seg = np.load(sys.argv[1]); M = json.load(open(sys.argv[2])); root, out = sys.argv[3], sys.argv[4]
st, sf = seg["times"], np.nan_to_num(seg["feats"].astype(np.float32))
def src_path(name):
    return glob.glob(os.path.join(root, "**", name + ".*.mp4"), recursive=True)[0]
cache = {}
def src_frames(name, t0, t1):
    c = av.open(src_path(name)); vs = c.streams.video[0]; vs.thread_type = "AUTO"
    c.seek(int(max(0, t0 - 1) / vs.time_base), stream=vs, backward=True)
    T, F = [], []
    for fr in c.decode(vs):
        t = float(fr.time)
        if t < t0: continue
        if t > t1: break
        T.append(t); F.append(feat(fr))
    c.close(); return np.array(T), np.nan_to_num(np.array(F, np.float32))
for r in M["shots"]:
    if not r.get("source"): continue
    seg_idx = np.where((st >= r["seg_start"] - 1e-3) & (st < r["seg_end"] - 1e-3))[0]
    T, F = src_frames(r["source"], r["src_start"] - 0.7, r["src_end"] + 0.7 + (r["seg_end"] - r["seg_start"]))
    if len(T) == 0: continue
    best = (-1, 0.0, r["rate"])
    for rate in sorted({r["rate"], 1.0, 0.0}):
      for d in np.arange(-0.6, 0.6001, 1 / 60):
        want = r["src_start"] + d + rate * (st[seg_idx] - r["seg_start"])
        j = np.clip(np.searchsorted(T, want), 1, len(T) - 1)
        j = np.where(np.abs(T[j - 1] - want) < np.abs(T[j] - want), j - 1, j)
        ok = np.abs(T[j] - want) < 0.03
        if ok.sum() < max(2, 0.6 * len(seg_idx)): continue
        sim = float(np.mean(np.sum(sf[seg_idx][ok] * F[j][ok], axis=1)))
        if sim > best[0] + 0.002: best = (sim, float(d), rate)
    if best[0] > 0:
        r["rate"] = best[2]
        r["src_start"] = round(r["src_start"] + best[1], 3); r["src_end"] = round(r["src_start"] + best[2] * (r["seg_end"] - r["seg_start"]), 3)
        r["refined_sim"] = round(best[0], 3); r["refine_shift_s"] = round(best[1], 3)
json.dump(M, open(out, "w"), indent=1)
for r in M["shots"]:
    if r.get("source"):
        print("  %02d -> %s %8.3f-%8.3f rate %.2f shift %+.3f sim %.3f" % (r["shot"], r["source"], r["src_start"], r["src_end"], r["rate"], r.get("refine_shift_s", 0), r.get("refined_sim", 0)))
