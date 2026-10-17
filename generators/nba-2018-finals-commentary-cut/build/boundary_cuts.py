"""Find hard cuts near the chapter edges, at full frame rate. Usage: boundary_cuts.py VIDEO T0 T1 [T0 T1 ...]"""
import sys, av, numpy as np
vpath = sys.argv[1]; spans = [(float(a), float(b)) for a, b in zip(sys.argv[2::2], sys.argv[3::2])]
c = av.open(vpath); vs = c.streams.video[0]; vs.thread_type = "AUTO"
print("fps", float(vs.average_rate))
for t0, t1 in spans:
    c.seek(int((t0 - 2) / vs.time_base), stream=vs, backward=True)
    prev = None; rows = []
    for fr in c.decode(vs):
        t = float(fr.time)
        if t < t0: continue
        if t > t1: break
        g = fr.to_ndarray(format="gray", width=64, height=36).astype(np.float32)
        if prev is not None: rows.append((t, float(np.mean(np.abs(g - prev))), float(g.mean())))
        prev = g
    rows.sort(key=lambda r: -r[1])
    print("span %.1f-%.1f top cuts:" % (t0, t1), ", ".join("%.3fs (diff %.0f, luma %.0f)" % r for r in rows[:4]))
