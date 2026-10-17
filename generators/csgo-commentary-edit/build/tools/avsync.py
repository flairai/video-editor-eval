"""Measure a clip's own audio-video sync from gunshots.

Every shot the observed player fires makes a muzzle flash next to the weapon and
a sharp high-frequency onset in the audio. Correlating the two event trains over
a lag window gives the audio's offset relative to the picture:
positive = audio late.

  python3 -E -P build/tools/avsync.py <video> <audio> [start_s end_s]
"""

import sys
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from media import load_audio  # noqa: E402

RATE = 16000


def flash_series(path, start=None, end=None):
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if start:
        container.seek(int(max(0.0, start - 1) / stream.time_base), stream=stream, backward=True)
    times, values, prev = [], [], None
    for frame in container.decode(stream):
        t = float(frame.time)
        if start is not None and t < start:
            continue
        if end is not None and t >= end:
            break
        g = frame.to_ndarray(format="gray", width=160, height=90).astype(np.float32)
        region = g[40:86, 72:150]  # lower-right half, where the weapon and muzzle flash sit
        top = float(np.mean(np.sort(region.reshape(-1))[-60:]))
        values.append(0.0 if prev is None else max(0.0, top - prev))
        prev = top
        times.append(t)
    container.close()
    v = np.array(values)
    v = np.maximum(0, (v - np.median(v)) / (np.std(v) + 1e-6) - 1.0)
    return np.array(times), v


def onset_series(path, start=None, end=None):
    audio, t0 = load_audio(path, RATE)
    hop, n = 16, 256  # 1 ms hop
    frames = np.lib.stride_tricks.sliding_window_view(audio, n)[::hop] * np.hanning(n)
    spec = np.abs(np.fft.rfft(frames, axis=1))[:, 32:128]  # about 2 to 8 kHz
    energy = np.log1p(spec.sum(axis=1))
    flux = np.maximum(0, np.diff(energy, prepend=energy[0]))
    times = t0 + (np.arange(len(flux)) * hop + n / 2) / RATE
    keep = np.ones_like(times, bool)
    if start is not None:
        keep &= times >= start - 0.5
    if end is not None:
        keep &= times < end + 0.5
    times, flux = times[keep], flux[keep]
    flux = np.maximum(0, (flux - np.median(flux)) / (np.std(flux) + 1e-6) - 1.0)
    kernel = np.exp(-0.5 * (np.arange(-24, 25) / 8.0) ** 2)
    return times, np.convolve(flux, kernel / kernel.sum(), mode="same")


def measure(video, audio, start=None, end=None, window_ms=300):
    vt, vs = flash_series(video, start, end)
    at, as_ = onset_series(audio, start, end)
    lags = np.arange(-window_ms, window_ms + 1)
    scores = np.array([np.sum(vs * np.interp(vt + lag / 1000.0, at, as_, left=0, right=0)) for lag in lags])
    best = int(np.argmax(scores))
    background = np.median(scores) + 1e-9
    return int(lags[best]), float(scores[best] / background), int(np.sum(vs > 0))


if __name__ == "__main__":
    args = sys.argv[1:]
    start, end = (float(args[2]), float(args[3])) if len(args) >= 4 else (None, None)
    lag, contrast, flashes = measure(args[0], args[1], start, end)
    print("audio offset %+d ms (contrast %.1f, %d flash frames)" % (lag, contrast, flashes))
