"""The images the judges see of a reconstruction, and the report shows: the recovered path against the
requested one, the fused point cloud from above and in overview (real colours, one colour per
frame, disagreement), the worst neighbouring pair and, for loops, the first frame against the last.
"""

from __future__ import annotations

import io
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FRAME_COLOURS = np.array([[230, 25, 75], [245, 130, 48], [255, 205, 0], [60, 180, 75], [70, 240, 240],
                          [0, 130, 200], [145, 30, 180], [240, 50, 230], [128, 128, 128]], np.uint8)


@lru_cache(maxsize=8)
def font(size: int, bold: bool = False):
    """DejaVu Sans from matplotlib's own files, so the images look the same on any machine."""
    try:
        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
        return ImageFont.truetype(path, size)
    except (OSError, ValueError):
        return ImageFont.load_default()


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def heat(err: np.ndarray) -> np.ndarray:
    """Error 0 green to 0.4 and above red; unseen (NaN) dark grey."""
    out = np.zeros(err.shape + (3,), np.uint8)
    e = np.nan_to_num(err, nan=-1)
    out[e < 0] = (40, 40, 40)
    val = np.clip(e / 0.4, 0, 1)
    m = e >= 0
    out[m] = np.stack([255 * val[m], 255 * (1 - val[m]) * 0.8, 40 * (1 - val[m])], -1).astype(np.uint8)
    return out


def point_heat(errs: np.ndarray) -> np.ndarray:
    v = np.clip(errs / 0.4, 0, 1)
    return np.stack([255 * v, 200 * (1 - v), 60 * np.ones_like(v)], -1).astype(np.uint8)


def render(P, colours, Ca, Rw, Cq, Rq, view: str, crop: dict, title: str, size=(960, 720)) -> Image.Image:
    """'top': orthographic floor plan from above; 'overview': perspective from behind and above the start."""
    Wd, Ht = size
    img = np.full((Ht, Wd, 3), 250, np.uint8)
    lo, hi = Cq.min(0) - crop["margin"], Cq.max(0) + crop["margin"]
    centre = (lo + hi) / 2
    if view == "top":
        span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        k = 0.92 * min(Wd, Ht) / span

        def proj(X):
            return np.stack([Wd / 2 + (X[:, 0] - centre[0]) * k, Ht / 2 - (X[:, 1] - centre[1]) * k], -1), X[:, 2]
    else:
        f0 = Rq[0][:, 2].copy()
        f0[2] = 0
        f0 /= (np.linalg.norm(f0) or 1)
        ext = float(np.linalg.norm(hi - lo))
        target = np.array([centre[0], centre[1], min(centre[2], Cq[:, 2].mean())])
        eye = target - 0.75 * ext * f0 + np.array([0, 0, 0.55 * ext])
        fwd = (target - eye) / np.linalg.norm(target - eye)
        r = np.cross(fwd, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        d = np.cross(fwd, r)
        Rv = np.stack([r, d, fwd])
        fl = 0.8 * Wd

        def proj(X):
            Xc = (X - eye) @ Rv.T
            z = np.maximum(Xc[:, 2], 1e-3)
            return np.stack([Wd / 2 + fl * Xc[:, 0] / z, Ht / 2 + fl * Xc[:, 1] / z], -1), -Xc[:, 2]
    if len(P):
        uv, depth_key = proj(P)
        order = np.argsort(depth_key)
        ui, vi = np.round(uv[order, 0]).astype(int), np.round(uv[order, 1]).astype(int)
        c = colours[order]
        for du in (0, 1):
            for dv in (0, 1):
                ok = (ui + du >= 0) & (ui + du < Wd) & (vi + dv >= 0) & (vi + dv < Ht)
                img[vi[ok] + dv, ui[ok] + du] = c[ok]
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    scale = float(np.linalg.norm(hi - lo)) * 0.05
    rq, _ = proj(Cq)
    dr.line([tuple(p) for p in rq], fill=(150, 150, 150), width=3)
    for p in rq:
        dr.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], outline=(120, 120, 120), width=2)
    rq_tip, _ = proj(Cq + Rq[:, :, 2] * scale)
    for p, q in zip(rq, rq_tip):
        dr.line([tuple(p), tuple(q)], fill=(150, 150, 150), width=2)
    cc, _ = proj(Ca)
    tips, _ = proj(Ca + Rw[:, :, 2] * scale * 1.4)
    dr.line([tuple(p) for p in cc], fill=(20, 20, 20), width=2)
    for k, (p, q) in enumerate(zip(cc, tips)):
        col = tuple(int(x) for x in FRAME_COLOURS[k % len(FRAME_COLOURS)])
        dr.line([tuple(p), tuple(q)], fill=col, width=4)
        dr.ellipse([p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9], fill=col, outline=(0, 0, 0), width=2)
        dr.text((p[0] + 11, p[1] - 12), str(k + 1), fill=(0, 0, 0), font=font(18, True))
    dr.text((12, 10), title, fill=(0, 0, 0), font=font(18, True))
    dr.text((12, Ht - 26), "grey: requested cameras and where they look  ·  numbered dots: recovered cameras, line = where each looks",
            fill=(60, 60, 60), font=font(14))
    return im


def path_figure(pc: dict, title: str, closure, tol: dict) -> Image.Image:
    v = pc["values"]
    Cq, Ca = pc["requested_centres"], pc["aligned_centres"]
    Rq, Rw = pc["requested_rot"], pc["aligned_rot"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=100, gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    span = float(np.ptp(Cq[:, :2], 0).max()) or 1.0
    for C, R, col, lab in ((Cq, Rq, "#9a9a9a", "requested"), (Ca, Rw, "#1f5fbf", "recovered")):
        ax.plot(C[:, 0], C[:, 1], "-", color=col, lw=2, label=lab)
        f = R[:, :, 2]
        ax.quiver(C[:, 0], C[:, 1], f[:, 0], f[:, 1], color=col, angles="xy", scale_units="xy", scale=1 / (0.12 * span), width=0.004)
        for k, (x, y) in enumerate(C[:, :2]):
            ax.annotate(str(k + 1), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9, color=col)
    ax.set_aspect("equal")
    ax.set_title("camera positions from above, arrows = where each looks")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    steps = np.arange(1, len(v["step_rotation_error_deg"]) + 1)
    ax.bar(steps - 0.18, v["step_rotation_error_deg"], width=0.36, color="#1f5fbf", label="rotation error (deg)")
    de = [np.nan if x is None else x for x in v["step_direction_error_deg"]]
    ax.bar(steps + 0.18, de, width=0.36, color="#e07b39", label="direction error (deg)")
    ax.axhline(tol["step_rotation_deg"], color="#1f5fbf", lw=1, ls=":")
    ax.axhline(tol["step_direction_deg"], color="#e07b39", lw=1, ls=":")
    ax.set_xticks(steps, ["%d→%d" % (k, k + 1) for k in steps], fontsize=8)
    ax.set_ylim(0, 180)
    ax.set_title("error of each step against the requested step (dotted: tolerance)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    extra = ""
    if closure:
        extra = " · loop closure: %.1f%% of the path length apart, %.1f° turned" % (
            100 * v["closure_distance_over_length"], v["closure_rotation_deg"])
    fig.suptitle("%s: recovered camera path vs requested%s" % (title, extra), fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return im


def worst_pair_figure(rec: dict, maps: dict, pair: dict) -> Image.Image:
    a, b = pair["from"] - 1, pair["to"] - 1
    err, warped = maps[(a, b)]
    target = rec["images"][b]
    H, W = err.shape
    out = Image.new("RGB", (3 * W + 40, H + 50), (255, 255, 255))
    dr = ImageDraw.Draw(out)
    for k, (arr, cap) in enumerate(((target, "cell %d as generated" % (b + 1)),
                                    (warped, "cell %d moved into cell %d's view" % (a + 1, b + 1)),
                                    (heat(err), "disagreement (green ok, red bad, grey unseen)"))):
        out.paste(Image.fromarray(arr), (k * (W + 20), 40))
        dr.text((k * (W + 20) + 4, 12), cap, fill=(0, 0, 0), font=font(16))
    return out


def closure_figure(rec: dict, maps: dict, closure) -> Image.Image:
    a, b = closure[0] - 1, closure[1] - 1
    err, warped = maps[(a, b)]
    imgs = rec["images"]
    H, W = err.shape
    out = Image.new("RGB", (4 * W + 60, H + 50), (255, 255, 255))
    dr = ImageDraw.Draw(out)
    for k, (arr, cap) in enumerate(((imgs[a], "cell %d" % (a + 1)), (imgs[b], "cell %d" % (b + 1)),
                                    (warped, "cell %d moved into cell %d's view" % (a + 1, b + 1)),
                                    (heat(err), "disagreement (green ok, red bad)"))):
        out.paste(Image.fromarray(arr), (k * (W + 20), 40))
        dr.text((k * (W + 20) + 4, 12), cap, fill=(0, 0, 0), font=font(16))
    return out
