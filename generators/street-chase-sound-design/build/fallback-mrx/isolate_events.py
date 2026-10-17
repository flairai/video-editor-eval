"""Find gunshots and tire screeches in the effects stem and write per-class stems (video-aligned) plus per-event clips."""
import json, sys, wave, numpy as np
from PIL import Image
sfx_path, flashes_path, out = sys.argv[1:4]
with wave.open(sfx_path) as w:
    sr = w.getframerate(); ch = w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
mono = x.mean(axis=1); hop, n = 441, 2048; win = np.hanning(n)
fr = np.lib.stride_tricks.sliding_window_view(mono, n)[::hop] * win
S = np.abs(np.fft.rfft(fr, axis=1)); f = np.fft.rfftfreq(n, 1 / sr); t = (np.arange(len(S)) * hop + n / 2) / sr
# --- transients (spectral flux on log magnitude)
L = np.log1p(20 * S); flux = np.maximum(0, np.diff(L, axis=0)).sum(axis=1); z = (flux - np.median(flux)) / (flux.std() + 1e-9)
tt = t[1:]
peaks = [i for i in range(3, len(z) - 3) if z[i] == z[i - 3:i + 4].max() and z[i] > 3.0]
onsets = [(float(tt[i]), float(z[i])) for i in peaks]
flashes = json.load(open(flashes_path))
gun_windows = [(9.25, 12.0), (18.33, 21.7)]
guns = []
for fl in flashes:
    cand = [o for o in onsets if 0.0 <= o[0] - fl["flash_s"] <= 0.20]
    if cand: guns.append(min(cand, key=lambda o: o[0] - fl["flash_s"])[0])
# strong transients inside the gunfight shots that no flash explains (e.g. a shot heard before its flash)
for o in onsets:
    if any(a <= o[0] < b for a, b in gun_windows) and o[1] > 6 and all(abs(o[0] - g) > 0.12 for g in guns):
        guns.append(o[0])
guns = sorted(set(round(g, 3) for g in guns))
# --- tire screech: sustained, tonal, high-band energy
band = (f >= 1000) & (f <= 6000); B = S[:, band]
peaky = B.max(axis=1) / (B.mean(axis=1) + 1e-9); energy = 20 * np.log10(np.sqrt((B ** 2).mean(axis=1)) + 1e-9)
e_thr = np.percentile(energy, 60); squeal = (peaky > 14) & (energy > e_thr)
segs = []; start = None
for i, s in enumerate(squeal):
    if s and start is None: start = i
    if (not s or i == len(squeal) - 1) and start is not None:
        if (i - start) * hop / sr >= 0.15: segs.append([float(t[start]), float(t[i])])
        start = None
merged = []
for s in segs:
    if merged and s[0] - merged[-1][1] < 0.25: merged[-1][1] = s[1]
    else: merged.append(s)
def gate(spans, pre, post):
    g = np.zeros(len(x), dtype=np.float32); fade = int(0.01 * sr)
    for a, b in spans:
        i0, i1 = max(0, int((a - pre) * sr)), min(len(x), int((b + post) * sr))
        g[i0:i1] = 1.0
        g[i0:i0 + fade] *= np.linspace(0, 1, min(fade, i1 - i0)); g[i1 - fade:i1] *= np.linspace(1, 0, min(fade, i1 - i0))
    return x * g[:, None]
def write(path, y):
    y = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes())
gun_spans = [(g, g) for g in guns]
write(out + "/gunshots.wav", gate(gun_spans, 0.02, 0.40))
write(out + "/tire_screech.wav", gate(merged, 0.03, 0.05))
for k, g in enumerate(guns, 1):
    i0, i1 = int((g - 0.05) * sr), int((g + 0.6) * sr); write("%s/clips/gunshot_%02d_%05.2fs.wav" % (out, k, g), x[i0:i1])
for k, (a, b) in enumerate(merged, 1):
    i0, i1 = int((a - 0.05) * sr), int((b + 0.1) * sr); write("%s/clips/tire_screech_%02d_%05.2fs.wav" % (out, k, a), x[i0:i1])
json.dump({"gunshot_onsets_s": guns, "tire_screech_spans_s": [[round(a, 2), round(b, 2)] for a, b in merged]}, open(out + "/events.json", "w"), indent=1)
# spectrogram image of the effects stem with marks
img = (np.clip((L.T[::-1][-300:] if False else L.T[::-1]) / L.max(), 0, 1) * 255).astype(np.uint8)
im = Image.fromarray(img).resize((1800, 400)).convert("RGB"); px = im.load()
for g in guns:
    xx = int(g / t[-1] * 1799)
    for yy in range(0, 30): px[xx, yy] = (255, 60, 60)
for a, b in merged:
    for xx in range(int(a / t[-1] * 1799), int(b / t[-1] * 1799)):
        for yy in range(30, 45): px[xx, yy] = (60, 200, 255)
im.save(out + "/sfx_spectrogram_marked.png")
print("gunshots (%d):" % len(guns), guns); print("tire screech spans (%d):" % len(merged), [[round(a, 2), round(b, 2)] for a, b in merged])
