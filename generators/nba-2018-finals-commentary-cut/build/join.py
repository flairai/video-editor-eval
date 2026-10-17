"""Join narration sentences to shots and their source footage.
Usage: join.py TRANSCRIPT.json MATCHES.json ANNOTATIONS.json SOURCES.json SEG_OFFSET OUT.json"""
import sys, json
tr, M, ann, srcmeta, off, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), json.load(open(sys.argv[4])), float(sys.argv[5]), sys.argv[6]
if isinstance(srcmeta.get("sources"), list):
    srcmeta = {e["id"]: e for e in srcmeta["sources"]}
FIX = {"tree": "streak", "3P.": "three-peat."}
words = []
for s in tr["segments"]:
    for w in s["words"]:
        txt = w["w"]
        if txt in FIX: txt = FIX[txt]
        words.append({"w": txt, "start": w["start"], "end": w["end"]})
sents, cur = [], []
for w in words:
    cur.append(w)
    if w["w"].endswith((".", "?", "!")) and w["w"] not in ("J.R.", ".R.", "J."):
        sents.append(cur); cur = []
if cur: sents.append(cur)
shots = M["shots"]
for r in shots:
    a = ann.get(str(r["shot"]))
    if a: r.update({"annotation": a})
def yt(t):
    t += off; return "%d:%05.2f" % (int(t) // 60, t % 60)
rows = []
for i, ws in enumerate(sents):
    s0, s1 = ws[0]["start"], ws[-1]["end"]
    hit = [r for r in shots if r["seg_start"] < s1 and r["seg_end"] > s0]
    text = ""
    for w in ws:
        t = w["w"]
        text += t if (not text or t.startswith(("-", "'", ".")) or t in (",", ".", "%")) else " " + t
    rows.append({"sentence": i, "text": text, "seg_start": s0, "seg_end": s1,
                 "youtube_start": yt(s0), "youtube_end": yt(s1),
                 "footage": [{"shot": r["shot"], "seg_start": r["seg_start"], "seg_end": r["seg_end"],
                              "source": r.get("source"), "src_start": r.get("src_start"), "src_end": r.get("src_end"),
                              "rate": r.get("rate"), "annotation": r.get("annotation")} for r in hit],
                 "words": ws})
quiet = []
for a, b in zip(sents, sents[1:]):
    if b[0]["start"] - a[-1]["end"] >= 2.0:
        quiet.append({"seg_start": a[-1]["end"], "seg_end": b[0]["start"]})
json.dump({"segment_offset_s": off, "sources": srcmeta, "sentences": rows, "no_narration_gaps": quiet, "shots": shots}, open(out, "w"), indent=1)
print("sentences", len(rows), "gaps >= 2 s:", [(round(q["seg_start"], 2), round(q["seg_end"], 2)) for q in quiet])
