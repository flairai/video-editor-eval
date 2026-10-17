"""Gunshots anchored on measured muzzle-flash matches; screeches found as 1-6 kHz energy bursts inside the four skid moments."""
import json, sys, wave, numpy as np
from PIL import Image
sfx_path, flash_path, out = sys.argv[1:4]
with wave.open(sfx_path) as w:
    sr, ch = w.getframerate(), w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
mono = x.mean(axis=1); hop, n = 441, 2048
fr = np.lib.stride_tricks.sliding_window_view(mono, n)[::hop] * np.hanning(n)
S = np.abs(np.fft.rfft(fr, axis=1)); f = np.fft.rfftfreq(n, 1 / sr); t = (np.arange(len(S)) * hop + n / 2) / sr
guns = sorted(r["audio_onset_s"] for r in json.load(open(flash_path)) if r["audio_onset_s"] is not None)
band = 10 * np.log10((S[:, (f > 1000) & (f < 6000)] ** 2).sum(axis=1) + 1e-12)
band_s = np.convolve(band, np.ones(5) / 5, mode="same")
moments = {"fishtail": (0.9, 2.4), "bus near-miss": (15.4, 17.9), "swerve": (21.6, 23.5), "peel-away": (28.1, 29.9)}
screech = []
for name, (a, b) in moments.items():
    sel = (t >= a) & (t <= b); loc = band_s[sel]; tt = t[sel]
    thr = np.median(band_s) + 0.6 * (loc.max() - np.median(band_s))
    on = loc > thr; runs = []; s0 = None
    for i, v in enumerate(on):
        if v and s0 is None: s0 = i
        if (not v or i == len(on) - 1) and s0 is not None:
            if tt[i] - tt[s0] >= 0.15: runs.append([float(tt[s0]), float(tt[i])])
            s0 = None
    if runs:
        span = max(runs, key=lambda r: r[1] - r[0]); screech.append({"moment": name, "start_s": round(span[0], 2), "end_s": round(span[1], 2)})
def gate(spans, pre, post):
    g = np.zeros(len(x), dtype=np.float32); fade = int(0.01 * sr)
    for a, b in spans:
        i0, i1 = max(0, int((a - pre) * sr)), min(len(x), int((b + post) * sr)); g[i0:i1] = 1.0
        k = min(fade, (i1 - i0) // 2); g[i0:i0 + k] *= np.linspace(0, 1, k); g[i1 - k:i1] *= np.linspace(1, 0, k)
    return x * g[:, None]
def write(path, y):
    y = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w: w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes())
write(out + "/gunshots.wav", gate([(g, g) for g in guns], 0.03, 0.40))
write(out + "/tire_screech.wav", gate([(s["start_s"], s["end_s"]) for s in screech], 0.05, 0.08))
for k, g in enumerate(guns, 1):
    write("%s/clips/gunshot_%02d_%05.2fs.wav" % (out, k, g), x[int((g - 0.05) * sr):int((g + 0.6) * sr)])
for k, s in enumerate(screech, 1):
    write("%s/clips/tire_screech_%02d_%05.2fs.wav" % (out, k, s["start_s"]), x[int((s["start_s"] - 0.05) * sr):int((s["end_s"] + 0.1) * sr)])
json.dump({"source": "MERL MRX default checkpoint, sfx stem", "gunshot_onsets_s": guns, "tire_screeches": screech}, open(out + "/events.json", "w"), indent=1)
# log-frequency spectrogram with marks
fl = np.geomspace(60, 16000, 220); idx = np.clip(np.searchsorted(f, fl), 0, len(f) - 1)
L = 20 * np.log10(S[:, idx] + 1e-6); L = np.clip((L - (L.max() - 70)) / 70, 0, 1)
im = Image.fromarray((L.T[::-1] * 255).astype(np.uint8)).resize((1800, 360)).convert("RGB"); px = im.load(); W = 1799 / t[-1]
for g in guns:
    for yy in range(0, 22): px[int(g * W), yy] = (255, 70, 70)
for s in screech:
    for xx in range(int(s["start_s"] * W), int(s["end_s"] * W)):
        for yy in range(24, 36): px[xx, yy] = (70, 200, 255)
im.save(out + "/sfx_spectrogram_marked.png")
print("gunshots (%d):" % len(guns), guns); print("screeches:", screech)
