"""Run MERL's MRX dialogue/music/effects separator on a 44.1 kHz WAV without torchaudio file I/O."""
import sys, wave, numpy as np, torch
repo, ckpt, src, out = sys.argv[1:5]
sys.path.insert(0, repo)
import separate
from mrx import MRX
def read(path):
    with wave.open(path) as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).T.astype(np.float32) / 32768.0
    assert sr == 44100, sr
    return torch.from_numpy(x.copy())
def write(path, t):
    x = np.clip(t.cpu().numpy().T, -1, 1); x = (x * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(44100); w.writeframes(x.tobytes())
model = MRX().eval(); model.load_state_dict(torch.load(ckpt, map_location="cpu"))
stems = separate.separate_soundtrack(read(src), separation_model=model, device=None, consistency_mode="pass", input_lufs=-27.0)
for name, t in stems.items():
    write("%s/%s.wav" % (out, name), t)
    rms = float(torch.sqrt(torch.mean(t ** 2)))
    print("%-6s rms %.4f (%.1f dBFS)" % (name, rms, 20 * np.log10(rms + 1e-9)))
