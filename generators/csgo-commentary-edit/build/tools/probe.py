"""Try candidate uploads for montage moments; print how well each matches.

  python3 -E -P build/tools/probe.py   (run from the case folder)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from media import frame_features, gcc_phat, load_audio  # noqa: E402

B = Path("build"); SR = 16000
MONTAGE_V = Path("sources/youtube") / "-jbMt_jpC7A.398.mp4"
PROBES = [
    ("M1 s1mple 'not FPL'", 6.93, 16.15, ["kUSN6u5CSRE", "E9RG_hkzqB4"]),
    ("M10 s1mple vs Na'Vi", 165.68, 172.62, ["kUSN6u5CSRE", "E9RG_hkzqB4"]),
    ("M11 chrisJ ace", 172.62, 190.95, ["rKRyCdXkj5w", "Hmc88Dnd4Fw", "iSpGQrtRw_Y", "ep8Q2uLlltQ"]),
    ("M12 Hiko inhuman", 190.95, 200.40, ["mXXw0XSQhes", "HnyRc6rbsK0", "qFGpLdGEL9k"]),
    ("M15 Hiko vs LG", 236.35, 255.00, ["dmrIfz1TN00", "z9i2wyhfOj8", "ceyzL5jrdOo"]),
    ("M16 rain quad", 255.00, 268.18, ["ev_oAGOYI7k"]),
    ("M17 NiKo deagle", 268.18, 278.80, ["QGFYLRKwrfA"]),
]
montage_audio, _ = load_audio(Path("sources/youtube") / "-jbMt_jpC7A.140.m4a", SR)
for label, ms, me, ids in PROBES:
    m_t, m_f = frame_features(MONTAGE_V, ms, me, every=3)
    for vid in ids:
        v = next(Path("sources/youtube").glob(vid + ".*.mp4")); a = next(Path("sources/youtube").glob(vid + ".*.m4a"))
        oa, _ = load_audio(a, SR)
        lag, peak, ratio = gcc_phat(oa, montage_audio[int(ms * SR): int(me * SR)], SR, len(oa) / SR)
        da = lag - ms
        o_t, o_f = frame_features(v)
        # global video search: best offset by matching every montage frame against all original frames
        sims = m_f @ o_f.T
        cand = o_t[np.argmax(sims, axis=1)] - m_t
        hist, edges = np.histogram(cand, bins=np.arange(cand.min() - 0.05, cand.max() + 0.1, 1 / 30))
        dv = float(edges[np.argmax(hist)] + 1 / 60)
        idx = np.clip(np.searchsorted(o_t, m_t + dv - 1 / 120), 0, len(o_t) - 1)
        s = np.einsum("ij,ij->i", m_f, o_f[idx]); s[(m_t + dv < 0) | (m_t + dv > o_t[-1])] = -1
        print("%-22s %s  audio %+8.2fs peak %.3f | video %+8.2fs median sim %.3f coverage %3.0f%% | skew %+7.0f ms"
              % (label, vid, da, peak, dv, float(np.median(s)), 100 * np.mean(s >= 0.85), (dv - da) * 1000))
