"""Frame features for footage matching: 64x36 grayscale, zero-mean, unit-norm."""
import numpy as np

W, H = 64, 36

def feat(frame):
    g = frame.to_ndarray(format="gray", width=W, height=H).astype(np.float32)
    g -= g.mean()
    n = np.linalg.norm(g)
    return (g / n if n > 1e-6 else g).reshape(-1)
