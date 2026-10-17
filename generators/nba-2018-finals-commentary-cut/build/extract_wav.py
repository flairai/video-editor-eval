"""Cut [start, end) of an audio file to 48 kHz stereo 16-bit WAV. Usage: extract_wav.py AUDIO START END OUT.wav"""
import sys, av, numpy as np
from fractions import Fraction
apath, start, end, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
ain = av.open(apath); ast = ain.streams.audio[0]
rs = av.AudioResampler(format="s16p", layout="stereo", rate=48000)
ain.seek(int(max(0, start - 3) / ast.time_base), stream=ast, backward=True)
chunks = []; t0 = None
for fr in ain.decode(ast):
    if t0 is None: t0 = float(fr.time)
    for r in rs.resample(fr): chunks.append(r.to_ndarray())
    if float(fr.time) > end + 1: break
pcm = np.concatenate(chunks, axis=1)
s0 = int(round((start - t0) * 48000)); s1 = s0 + int(round((end - start) * 48000))
pcm = np.ascontiguousarray(pcm[:, s0:s1])
oc = av.open(out, "w"); oa = oc.add_stream("pcm_s16le", rate=48000)
pts = 0
for k in range(0, pcm.shape[1], 4096):
    blk = np.ascontiguousarray(pcm[:, k:k + 4096])
    af = av.AudioFrame.from_ndarray(np.ascontiguousarray(blk.T.reshape(1, -1)), format="s16", layout="stereo"); af.sample_rate = 48000
    af.pts = pts; af.time_base = Fraction(1, 48000); pts += blk.shape[1]
    for p in oa.encode(af): oc.mux(p)
for p in oa.encode(): oc.mux(p)
oc.close(); print("samples", pcm.shape[1], "seconds", round(pcm.shape[1] / 48000, 4))
