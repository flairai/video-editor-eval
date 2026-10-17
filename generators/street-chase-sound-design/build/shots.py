"""Detect hard cuts and save one mid-shot frame per shot as 4x4 contact sheets.

    python3 build/shots.py sources/7DXP7HsnL2A.398.mp4 build/intermediate/reference-shots

Writes <outdir>/shots.json and <outdir>/shots_NN.jpg. A cut is a frame whose 64x36
grayscale difference from the previous frame is 8 MADs above the median and above 18.
"""

import json
import sys

import av
import numpy as np
from PIL import Image, ImageDraw


def main(video: str, outdir: str) -> None:
    with av.open(video) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        fps, prev, diffs, times = float(vs.average_rate), None, [], []
        for frame in c.decode(vs):
            g = frame.to_ndarray(format="gray", width=64, height=36).astype(np.float32)
            if prev is not None:
                diffs.append(np.mean(np.abs(g - prev)))
                times.append(float(frame.time))
            prev = g
    d, ts = np.array(diffs), np.array(times)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-6
    cuts = [0.0]
    for i in np.where((d > med + 8 * mad) & (d > 18))[0]:
        lo, hi = max(0, i - 2), min(len(d), i + 3)
        if d[i] == d[lo:hi].max() and ts[i] - cuts[-1] > 0.15:
            cuts.append(float(ts[i]))
    end = float(ts[-1]) + 1 / fps
    shots = [{"n": k + 1, "start": round(s, 3), "end": round(e, 3), "dur": round(e - s, 3)}
             for k, (s, e) in enumerate(zip(cuts, cuts[1:] + [end]))]
    json.dump({"fps": fps, "shots": shots}, open(outdir + "/shots.json", "w"), indent=1)
    tiles = []
    with av.open(video) as c:
        vs = c.streams.video[0]
        for s in shots:
            t = (s["start"] + s["end"]) / 2
            c.seek(int(t / vs.time_base), stream=vs, backward=True)
            for frame in c.decode(vs):
                if float(frame.time) >= t:
                    im = frame.to_image().resize((320, frame.height * 320 // frame.width))
                    draw = ImageDraw.Draw(im)
                    draw.rectangle((0, 0, 70, 16), fill="black")
                    draw.text((3, 2), "%d %.1fs" % (s["n"], t), fill="yellow")
                    tiles.append(im)
                    break
    w, h = tiles[0].size
    for start in range(0, len(tiles), 16):
        sheet = Image.new("RGB", (w * 4, h * 4), "black")
        for i, im in enumerate(tiles[start : start + 16]):
            sheet.paste(im.resize((w, h)), ((i % 4) * w, (i // 4) * h))
        sheet.save("%s/shots_%02d.jpg" % (outdir, start // 16), quality=80)
    print("fps %.3f, %d shots, mean %.2f s" % (fps, len(shots), np.mean([s["dur"] for s in shots])))


if __name__ == "__main__":
    main(*sys.argv[1:3])
