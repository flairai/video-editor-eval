"""Rows of consecutive numbered frames, optionally cropped, for pinning event frames by eye.

    python3 build/frame_rows.py $T/environment/workspace/graphite-run.mp4 build/intermediate/frames/<name>.jpg \
        '[{"label": "taxi", "from": 544, "to": 549, "crop": [500, 300, 1200, 700]}]'

Each row is {"label", "from", "to", "step" (default 1), "crop" ([x0, y0, x1, y1] in source
pixels, default the whole frame)}. Every tile is 320 px wide and captioned with its frame
number and time; frame f is on screen from f/fps seconds.
"""

import json
import sys

import av
from PIL import Image, ImageDraw


def main(video: str, out: str, rows_json: str) -> None:
    rows = json.loads(rows_json)
    want = {f for r in rows for f in range(r["from"], r["to"] + 1, r.get("step", 1))}
    got = {}
    with av.open(video) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate)
        for frame in c.decode(vs):
            k = round(float(frame.time) * fps)
            if k in want:
                got[k] = frame.to_image()
            if k > max(want):
                break
    width, tiles = 320, []
    for r in rows:
        x0, y0, x1, y1 = r.get("crop", [0, 0, *got[min(got)].size])
        height = round(width * (y1 - y0) / (x1 - x0))
        tiles.append([(k, got[k].crop((x0, y0, x1, y1)).resize((width, height)))
                      for k in range(r["from"], r["to"] + 1, r.get("step", 1)) if k in got])
    heights = [t[0][1].size[1] + 16 for t in tiles]
    sheet = Image.new("RGB", (width * max(len(t) for t in tiles), sum(heights)), "black")
    draw, y = ImageDraw.Draw(sheet), 0
    for r, t, h in zip(rows, tiles, heights):
        for j, (k, im) in enumerate(t):
            sheet.paste(im, (j * width, y + 16))
            draw.text((j * width + 3, y + 2), "%s f%d %.3fs" % (r["label"] if j == 0 else "", k, k / fps), fill="yellow")
        y += h
    sheet.save(out, quality=85)
    print(out, sheet.size)


if __name__ == "__main__":
    main(*sys.argv[1:4])
