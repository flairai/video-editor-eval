"""Cut [start, end) from a video-only + audio-only pair into one H.264/AAC mp4.
Usage: extract_segment.py VIDEO AUDIO START END OUT.mp4"""
import sys, av, numpy as np
from fractions import Fraction
vpath, apath, start, end, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
oc = av.open(out, "w")
vin = av.open(vpath); vs = vin.streams.video[0]; vs.thread_type = "AUTO"
fps = vs.average_rate
ov = oc.add_stream("libx264", rate=fps)
ov.width, ov.height, ov.pix_fmt = vs.codec_context.width, vs.codec_context.height, "yuv420p"
ov.options = {"crf": "16", "preset": "medium"}
oa = oc.add_stream("aac", rate=48000)
# video
vin.seek(int((start - 2) / vs.time_base), stream=vs, backward=True)
n = 0; first_t = None
for fr in vin.decode(vs):
    t = float(fr.time)
    if t < start - 1e-6: continue
    if t >= end - 1e-6: break
    if first_t is None: first_t = t
    f2 = fr.reformat(format="yuv420p"); f2.pts = n; f2.time_base = Fraction(fps.denominator, fps.numerator)
    for p in ov.encode(f2): oc.mux(p)
    n += 1
for p in ov.encode(): oc.mux(p)
# audio: decode, resample, cut by sample index from the first video frame time
ain = av.open(apath); ast = ain.streams.audio[0]
rs = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
ain.seek(int(max(0, first_t - 3) / ast.time_base), stream=ast, backward=True)
chunks = []; t_audio0 = None
for fr in ain.decode(ast):
    if t_audio0 is None: t_audio0 = float(fr.time)
    for r in rs.resample(fr): chunks.append(r.to_ndarray())
    if float(fr.time) > first_t + (n / float(fps)) + 1: break
pcm = np.concatenate(chunks, axis=1)
s0 = int(round((first_t - t_audio0) * 48000)); s1 = s0 + int(round(n / float(fps) * 48000))
pcm = np.ascontiguousarray(pcm[:, s0:s1], dtype=np.float32)
fifo_pts = 0
for i in range(0, pcm.shape[1], 1024):
    blk = np.ascontiguousarray(pcm[:, i:i + 1024])
    af = av.AudioFrame.from_ndarray(blk, format="fltp", layout="stereo"); af.sample_rate = 48000
    af.pts = fifo_pts; af.time_base = Fraction(1, 48000); fifo_pts += blk.shape[1]
    for p in oa.encode(af): oc.mux(p)
for p in oa.encode(): oc.mux(p)
oc.close()
print("frames", n, "first video t", round(first_t, 4), "audio samples", pcm.shape[1], "dur", round(n / float(fps), 3))
