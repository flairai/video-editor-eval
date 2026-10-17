"""Index a source video: features at ~10 fps plus exact frame times. Usage: index_sources.py VIDEO OUT.npz"""
import sys, av, numpy as np
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from vfeat import feat
vpath, out = sys.argv[1], sys.argv[2]
c = av.open(vpath); vs = c.streams.video[0]; vs.thread_type = "AUTO"
fps = float(vs.average_rate); step = max(1, round(fps / 10))
times, feats = [], []
for i, fr in enumerate(c.decode(vs)):
    if i % step:
        continue
    times.append(float(fr.time)); feats.append(feat(fr))
np.savez_compressed(out, times=np.array(times, np.float64), feats=np.array(feats, np.float16), fps=fps)
print(vpath.rsplit("/", 1)[-1], "frames", len(times), "fps", round(fps, 3), "dur", round(times[-1], 1))
