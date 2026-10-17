"""Render QA sheets. Usage: qa_sheet.py SEGMENT.mp4 MATCHES.json SRC_DIR OUT_PREFIX [unmatched|matched]"""
import sys, json, os, glob, av
from PIL import Image, ImageDraw
seg_path, mpath, src_dir, prefix, mode = sys.argv[1:6]
M = json.load(open(mpath))["shots"]
def frame_at(path, t, size=(426, 240)):
    c = av.open(path); vs = c.streams.video[0]
    c.seek(int(max(0, t - 1) / vs.time_base), stream=vs, backward=True)
    im = None
    for fr in c.decode(vs):
        if float(fr.time) >= t - 1e-3:
            im = fr.to_image().resize(size); break
    c.close(); return im or Image.new("RGB", size)
def label(im, text):
    d = ImageDraw.Draw(im); d.rectangle((0, 0, im.width, 18), fill=(0, 0, 0)); d.text((4, 3), text, fill=(255, 255, 0)); return im
def src_file(name):
    return glob.glob(os.path.join(src_dir, "**", name + ".*.mp4"), recursive=True)[0]
rows = [r for r in M if (r["source"] is None) == (mode == "unmatched")]
if mode == "unmatched":
    per = 12
    for p in range(0, len(rows), per):
        sheet = Image.new("RGB", (426 * 4, 240 * 3))
        for i, r in enumerate(rows[p:p + per]):
            mid = (r["seg_start"] + r["seg_end"]) / 2
            sheet.paste(label(frame_at(seg_path, mid), "shot %02d  %.1f-%.1fs" % (r["shot"], r["seg_start"], r["seg_end"])), ((i % 4) * 426, (i // 4) * 240))
        sheet.save("%s_unmatched_%d.jpg" % (prefix, p // per), quality=85)
else:
    per = 6
    for p in range(0, len(rows), per):
        sheet = Image.new("RGB", (852, 240 * per))
        for i, r in enumerate(rows[p:p + per]):
            mid = (r["seg_start"] + r["seg_end"]) / 2
            st = r["src_start"] + r["rate"] * (mid - r["seg_start"])
            sheet.paste(label(frame_at(seg_path, mid), "seg shot %02d @%.2fs" % (r["shot"], mid)), (0, i * 240))
            sheet.paste(label(frame_at(src_file(r["source"]), st), "%s @%.2fs" % (r["source"], st)), (426, i * 240))
        sheet.save("%s_matched_%d.jpg" % (prefix, p // per), quality=85)
print("rows", len(rows))
