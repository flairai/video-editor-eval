"""Small PyAV helpers shared by the build tools and the scorer."""

import av
import numpy as np


def load_audio(path, rate, layout="mono"):
    """Decode a whole audio track to float32 at `rate`. Returns (samples, start_time_s)."""
    container = av.open(str(path))
    stream = container.streams.audio[0]
    # Planar float keeps channels separate: to_ndarray() is (channels, samples).
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
    chunks, t0 = [], None
    for frame in container.decode(stream):
        if t0 is None and frame.time is not None:
            t0 = float(frame.time)
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray())
    for out in resampler.resample(None):
        chunks.append(out.to_ndarray())
    container.close()
    channels = 1 if layout == "mono" else 2
    data = np.concatenate(chunks, axis=1) if chunks else np.zeros((channels, 0), np.float32)
    data = data.reshape(channels, -1) if layout != "mono" else data.reshape(-1)
    return data.astype(np.float32), (t0 or 0.0)


def video_info(path):
    container = av.open(str(path))
    stream = container.streams.video[0]
    info = {"fps": float(stream.average_rate), "width": stream.width, "height": stream.height,
            "start_s": float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0,
            "duration_s": float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / 1e6}
    container.close()
    return info


def frame_features(path, start=None, end=None, every=1, size=(64, 36)):
    """Gray thumbnails, mean-removed and L2-normalised, for frames in [start, end). Returns (times, feats)."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if start:
        container.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream, backward=True)
    times, feats, count = [], [], 0
    for frame in container.decode(stream):
        t = float(frame.time)
        if start is not None and t < start - 1e-6:
            continue
        if end is not None and t >= end:
            break
        if count % every == 0:
            g = frame.to_ndarray(format="gray", width=size[0], height=size[1]).astype(np.float32).reshape(-1)
            g -= g.mean()
            n = np.linalg.norm(g)
            feats.append(g / n if n > 1e-3 else g)
            times.append(t)
        count += 1
    container.close()
    return np.array(times), (np.stack(feats) if feats else np.zeros((0, size[0] * size[1]), np.float32))


def gcc_phat(signal, reference, rate, max_shift_s):
    """Lag (seconds) of `reference` inside `signal` by generalised cross-correlation with phase transform."""
    n = len(signal) + len(reference)
    size = 1 << int(np.ceil(np.log2(n)))
    spec = np.fft.rfft(signal, size) * np.conj(np.fft.rfft(reference, size))
    spec /= np.abs(spec) + 1e-12
    corr = np.fft.irfft(spec, size)
    max_shift = int(max_shift_s * rate)
    lags = np.concatenate([np.arange(0, max_shift + 1), np.arange(-max_shift, 0)])
    values = corr[lags % size]
    best = int(np.argmax(values))
    ranked = np.sort(values)[::-1]
    return lags[best] / rate, float(ranked[0]), float(ranked[0] / (ranked[1] + 1e-12))
