"""Mux a separate video download and audio download into one H.264 + AAC MP4 for upload.

    python3 build/transcode_proxy.py sources/7DXP7HsnL2A.398.mp4 sources/7DXP7HsnL2A.140.m4a \
        build/intermediate/infinite-chase.mp4 h264_videotoolbox

The YouTube video is AV1, which PyAV decodes but cannot stream-copy into MP4, so it is
re-encoded (h264_videotoolbox on macOS, libx264 elsewhere) with explicit timestamps.
"""

import sys
from fractions import Fraction

import av


def main(video: str, audio: str, out: str, encoder: str) -> None:
    with av.open(video) as v, av.open(audio) as a, av.open(out, "w", format="mp4") as o:
        vin = v.streams.video[0]
        rate = vin.average_rate
        vout = o.add_stream(encoder, rate=rate)
        vout.width, vout.height, vout.pix_fmt = vin.codec_context.width, vin.codec_context.height, "yuv420p"
        vout.bit_rate = 6_000_000
        vout.codec_context.time_base = Fraction(1) / rate
        aout = o.add_stream("aac", rate=48000)
        aout.layout = "stereo"
        frames = 0
        for frame in v.decode(vin):
            frame = frame.reformat(format="yuv420p")
            frame.pts, frame.time_base = frames, vout.codec_context.time_base
            frames += 1
            for packet in vout.encode(frame):
                o.mux(packet)
        for packet in vout.encode():
            o.mux(packet)
        resampler, samples = av.AudioResampler(format="fltp", layout="stereo", rate=48000), 0
        for frame in a.decode(a.streams.audio[0]):
            for r in resampler.resample(frame):
                r.pts, r.time_base = samples, Fraction(1, 48000)
                samples += r.samples
                for packet in aout.encode(r):
                    o.mux(packet)
        for packet in aout.encode():
            o.mux(packet)
    print(encoder, frames, "frames ->", out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
