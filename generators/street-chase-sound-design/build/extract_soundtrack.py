"""Write a video's soundtrack as a 16-bit stereo WAV at the given sample rate.

    python3 build/extract_soundtrack.py sources/graphite-run-v1.mp4 build/intermediate/qc/soundtrack_44k.wav 44100
"""

import sys
import wave

import av
import numpy as np


def main(video: str, out: str, rate: str) -> None:
    with av.open(video) as c:
        resampler = av.AudioResampler(format="s16", layout="stereo", rate=int(rate))
        pcm = np.concatenate([r.to_ndarray().reshape(-1) for f in c.decode(c.streams.audio[0]) for r in resampler.resample(f)])
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.astype(np.int16).tobytes())
    print(out, round(len(pcm) / 2 / int(rate), 3), "s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
