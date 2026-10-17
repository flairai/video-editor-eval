"""Render the alignment JSON as a readable Markdown table. Usage: make_readme.py ALIGNMENT.json OUT.md"""
import sys, json
A = json.load(open(sys.argv[1])); out = sys.argv[2]
off = A["segment_offset_s"]; src = A["sources"]
SHORT = {"fA0GE3WL2yc": "Finals G1 highlights", "ErdrnfPCe7c": "Finals G2 highlights", "vATsPbgNnp8": "Finals G3 highlights",
         "S3QMhrPo0gg": "Finals G4 highlights", "qmDsgGKHBEo": "ESPN G1 final sequence", "RgB5Ub1GNUU": "LeBron 51 (G1)",
         "dek9OSrjHLg": "Durant 43 (G3)", "N-SPJ79WNAg": "2018 ECSF G2 highlights", "4umRHTpWLsg": "LeBron ECSF G2 (NBA)",
         "Htds3dbB-wY": "2013 Finals G7, LeBron 37", "r_liBadmQtQ": "LeBron forces G7 (ECF G6)", "8XiWmC0V8ww": "ECF G6 LeBron highlights",
         "gd-at0w1VGA": "ESPN best of ECF G6", "LsFwdCgDlbQ": "LeBron checks out (G4)", "XA_-eFS3eVI": "ESPN standing ovation (G4)",
         "cLpIPP9DZic": "ESPN Durant MVP speech", "XTsncSDMSpg": "ESPN trophy presentation", "2cX-OUdxU0s": "Finals G1 intro",
         "vG-9l7PS_4U": "Durant best access (celebration)"}
def ts(t):
    return "%d:%05.2f" % (int(t) // 60, t % 60)
def foot(f):
    if f.get("source"):
        sp = "" if abs((f.get("rate") or 1) - 1) < 0.05 else (" (freeze)" if (f.get("rate") or 0) < 0.05 else " (%.2fx)" % f["rate"])
        return "%s %s to %s%s" % (SHORT.get(f["source"], f["source"]), ts(f["src_start"]), ts(f["src_end"]), sp)
    a = f.get("annotation")
    return ("*%s*" % a["desc"]) if a else "*no match yet*"
shots = A["shots"]
m = [s for s in shots if s.get("source")]; g = [s for s in shots if not s.get("source") and (s.get("annotation") or {}).get("type") == "graphic"]
dur = shots[-1]["seg_end"]
L = []
L.append("# Daniel Li's 2018 Finals commentary, matched to the original footage\n")
L.append("Reference: [What Actually Happened In Every NBA Finals Since 2000 (Part 2)](https://www.youtube.com/watch?v=caVtD4IQhmk), chapter \"2018 Finals\".\n")
L.append("- Exact bounds: %s to %s in the YouTube video (%.3f s to %.3f s), %.1f s, %d shots. The start is the cut into the 2018 title card; the end is the cut into the 2019 title card." % (ts(off), ts(off + dur), off, off + dur, dur, len(shots)))
L.append("- Matched to original footage: %d shots, %.1f s of %.1f s. Graphics made by Daniel Li (title and tweet cards): %d shots, %.1f s. Unmatched footage: %d shots, %.1f s." % (
    len(m), sum(s["seg_end"] - s["seg_start"] for s in m), dur, len(g), sum(s["seg_end"] - s["seg_start"] for s in g),
    len(shots) - len(m) - len(g), dur - sum(s["seg_end"] - s["seg_start"] for s in m) - sum(s["seg_end"] - s["seg_start"] for s in g)))
L.append("- Narration-free stretches of 2 s or more: " + ", ".join("%s to %s" % (ts(off + q["seg_start"]), ts(off + q["seg_end"])) for q in A["no_narration_gaps"]) + ".\n")
L.append("## Commentary and the footage on screen\n")
L.append("Times in the first column are YouTube times in Daniel Li's video. Source times are positions inside each source video.\n")
L.append("| YouTube time | Narration | Footage on screen while it is said |")
L.append("|---|---|---|")
for r in A["sentences"]:
    seen, order = set(), []
    for f in sorted(r["footage"], key=lambda f: f["shot"]):
        x = foot(f)
        if x not in seen: seen.add(x); order.append(x)
    feet = "<br>".join(order)
    L.append("| %s to %s | %s | %s |" % (r["youtube_start"], r["youtube_end"], r["text"].replace("|", "/"), feet))
L.append("\n## Shot list\n")
L.append("| Shot | YouTube time | Source | Source time | Similarity |")
L.append("|---|---|---|---|---|")
for s in shots:
    if s.get("source"):
        L.append("| %02d | %s to %s | %s | %s to %s | %.3f |" % (s["shot"], ts(off + s["seg_start"]), ts(off + s["seg_end"]), SHORT.get(s["source"], s["source"]), ts(s["src_start"]), ts(s["src_end"]), s.get("refined_sim", s.get("mean_sim", 0))))
    else:
        a = s.get("annotation")
        L.append("| %02d | %s to %s | %s | | |" % (s["shot"], ts(off + s["seg_start"]), ts(off + s["seg_end"]), ("*" + a["desc"] + "*") if a else "*no match yet*"))
L.append("\n## Source videos\n")
L.append("| Id | Channel | Title |")
L.append("|---|---|---|")
for i, r in src.items():
    L.append("| [%s](%s) | %s | %s |" % (i, r["url"], r["channel"], r["title"].replace("|", "/")))
open(out, "w").write("\n".join(L) + "\n"); print("wrote", out)
