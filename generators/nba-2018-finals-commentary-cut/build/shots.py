"""Hard-cut detection at full frame rate, plus segment features at every frame.
Usage: shots.py SEGMENT.mp4 OUT.npz"""
import sys, av, numpy as np
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from vfeat import feat
vpath, out = sys.argv[1], sys.argv[2]
c = av.open(vpath); vs = c.streams.video[0]; vs.thread_type = "AUTO"
fps = float(vs.average_rate)
times, feats, thumbs, diffs = [], [], [], []
prev = None
for fr in c.decode(vs):
    g = fr.to_ndarray(format="gray", width=64, height=36).astype(np.float32)
    diffs.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
    prev = g
    times.append(float(fr.time)); feats.append(feat(fr))
d = np.array(diffs); t = np.array(times)
med = np.median(d); mad = np.median(np.abs(d - med)) + 1e-6
cand = [i for i in range(1, len(d)) if d[i] > max(18.0, med + 10 * mad) and d[i] == d[max(0, i - 2):i + 3].max()]
cuts = [0]
for i in cand:
    if i - cuts[-1] >= 4: cuts.append(i)
bounds = cuts + [len(t)]
shots = [(bounds[k], bounds[k + 1] - 1) for k in range(len(bounds) - 1)]
np.savez_compressed(out, times=t, feats=np.array(feats, np.float16), diffs=d, shots=np.array(shots), fps=fps)
print("frames", len(t), "fps", round(fps, 3), "shots", len(shots))
for k, (a, b) in enumerate(shots):
    print("  shot %02d  %7.3f-%7.3f  (%5.2f s)" % (k, t[a], t[b] + 1 / fps, (b - a + 1) / fps))
