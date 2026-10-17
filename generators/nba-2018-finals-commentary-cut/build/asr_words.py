"""Word-level transcript with faster-whisper. Usage: asr_words.py MEDIA OUT.json MODEL"""
import sys, json
from faster_whisper import WhisperModel
media, out, model = sys.argv[1], sys.argv[2], sys.argv[3]
m = WhisperModel(model, device="cpu", compute_type="int8")
segs, info = m.transcribe(media, language="en", beam_size=5, word_timestamps=True, vad_filter=False)
rows = []
for s in segs:
    rows.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                 "words": [{"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3), "p": round(w.probability, 3)} for w in s.words]})
json.dump({"model": model, "segments": rows}, open(out, "w"), indent=1)
for r in rows:
    print("[%6.2f-%6.2f] %s" % (r["start"], r["end"], r["text"]))
