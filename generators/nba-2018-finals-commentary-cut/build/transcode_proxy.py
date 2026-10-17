"""Transcode a video-only + audio-only pair to one 640x360 H.264/AAC mp4 at <=30 fps,
keeping source timestamps (second t in the source is second t in the proxy).
Usage: transcode_proxy.py VIDEO AUDIO OUT.mp4"""
import sys, av, numpy as np
from fractions import Fraction
vpath, apath, out = sys.argv[1], sys.argv[2], sys.argv[3]
vin = av.open(vpath); vs = vin.streams.video[0]; vs.thread_type = "AUTO"
src_rate = vs.average_rate
step = 2 if float(src_rate) > 31 else 1
rate = src_rate / step
oc = av.open(out, "w")
ov = oc.add_stream("libx264", rate=rate)
ov.width, ov.height, ov.pix_fmt = 640, 360, "yuv420p"
ov.options = {"crf": "26", "preset": "medium"}
tb = Fraction(rate.denominator, rate.numerator)
oa = oc.add_stream("aac", rate=48000)
n = 0; first = None; last_pts = -1
for i, fr in enumerate(vin.decode(vs)):
    if i % step: continue
    t = float(fr.time)
    if first is None: first = t
    pts = int(round(t / float(tb)))
    if pts <= last_pts: pts = last_pts + 1
    f2 = fr.reformat(width=640, height=360, format="yuv420p"); f2.pts = pts; f2.time_base = tb; last_pts = pts
    for p in ov.encode(f2): oc.mux(p)
    n += 1
for p in ov.encode(): oc.mux(p)
ain = av.open(apath); ast = ain.streams.audio[0]
rs = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
chunks = []; t0 = None
for fr in ain.decode(ast):
    if t0 is None: t0 = float(fr.time)
    for r in rs.resample(fr): chunks.append(r.to_ndarray())
pcm = np.concatenate(chunks, axis=1).astype(np.float32)
if t0 > 0: pcm = np.concatenate([np.zeros((2, int(round(t0 * 48000))), np.float32), pcm], axis=1)
elif t0 < 0: pcm = pcm[:, int(round(-t0 * 48000)):]
pts = 0
for k in range(0, pcm.shape[1], 1024):
    blk = np.ascontiguousarray(pcm[:, k:k + 1024])
    af = av.AudioFrame.from_ndarray(blk, format="fltp", layout="stereo"); af.sample_rate = 48000
    af.pts = pts; af.time_base = Fraction(1, 48000); pts += blk.shape[1]
    for p in oa.encode(af): oc.mux(p)
for p in oa.encode(): oc.mux(p)
oc.close()
print(out.rsplit("/", 1)[-1], "frames", n, "fps", round(float(rate), 3), "first_video_t", round(first, 3), "audio_t0", round(t0, 3), "audio_s", round(pcm.shape[1] / 48000, 2))
