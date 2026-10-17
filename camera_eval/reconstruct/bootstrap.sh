#!/usr/bin/env bash
# Set up a GPU host to serve VGGT-Omega (camera_eval/reconstruct/server.py). Safe to run again: each
# step checks its own result first. camera_eval.gpu runs it over SSH with these variables set:
#   CE_HOME       where everything goes (default ~/camera-eval)
#   CE_VGGT_SHA   facebookresearch/vggt-omega commit to install
#   CE_CKPT_REPO  Hugging Face repo of the checkpoint, CE_CKPT_FILE its file, CE_CKPT_SHA256 its hash
#   CE_ALLOW_NO_CUDA=1  accept a torch without CUDA (camera_eval.gpu.local on a Mac: MPS or CPU)
# Needs python3 >= 3.10 with pip or venv, and an NVIDIA GPU (torch >= 2.6 is installed when missing).
# No credential is needed or used: the vggt-omega source is public and the checkpoint is ungated.
set -euo pipefail
CE_HOME="${CE_HOME:-$HOME/camera-eval}"
: "${CE_VGGT_SHA:?}" "${CE_CKPT_REPO:?}" "${CE_CKPT_FILE:?}" "${CE_CKPT_SHA256:?}"
mkdir -p "$CE_HOME/ckpt" "$CE_HOME/logs"
cd "$CE_HOME"
log="$CE_HOME/logs/bootstrap.log"

# A venv that sees the system's packages, so an image that already has torch (the pytorch images,
# Deep Learning AMIs) keeps it instead of downloading another.
if [ ! -x "$CE_HOME/venv/bin/python" ]; then
  python3 -m venv --system-site-packages "$CE_HOME/venv" >>"$log" 2>&1 || {
    echo "python3 -m venv failed (on Debian/Ubuntu: apt-get install python3-venv); see $log"; exit 1; }
fi
PY="$CE_HOME/venv/bin/python"
"$PY" -m pip install -q --upgrade pip >>"$log" 2>&1 || true

if ! CE_ALLOW_NO_CUDA="${CE_ALLOW_NO_CUDA:-0}" "$PY" - <<'EOF' >>"$log" 2>&1
import os, sys, torch
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
sys.exit(0 if (major, minor) >= (2, 6) and (torch.cuda.is_available() or os.environ["CE_ALLOW_NO_CUDA"] == "1") else 1)
EOF
then
  echo "installing torch (none >= 2.6 with CUDA found)"
  "$PY" -m pip install -q "torch>=2.6" >>"$log" 2>&1 || { tail -20 "$log"; exit 1; }
fi

"$PY" -m pip install -q "numpy<2" einops safetensors pillow "huggingface_hub[cli]" hf_xet >>"$log" 2>&1 || { tail -20 "$log"; exit 1; }

src="$CE_HOME/vggt-omega-$CE_VGGT_SHA"
if [ ! -d "$src" ]; then
  "$PY" - "$CE_VGGT_SHA" "$CE_HOME" <<'EOF'
import io, sys, urllib.request, zipfile
sha, home = sys.argv[1], sys.argv[2]
data = urllib.request.urlopen("https://github.com/facebookresearch/vggt-omega/archive/%s.zip" % sha, timeout=300).read()
zipfile.ZipFile(io.BytesIO(data)).extractall(home)
EOF
fi
if ! "$PY" -c "import vggt_omega, os, sys; sys.exit(0 if os.path.realpath(vggt_omega.__path__[0]).startswith(os.path.realpath('$src')) else 1)" >>"$log" 2>&1; then
  "$PY" -m pip install -q --no-deps -e "$src" >>"$log" 2>&1 || { tail -20 "$log"; exit 1; }
fi

ckpt="$CE_HOME/ckpt/$CE_CKPT_FILE"
if [ ! -f "$ckpt.ok" ]; then
  if [ ! -f "$ckpt" ]; then
    "$CE_HOME/venv/bin/hf" download "$CE_CKPT_REPO" "$CE_CKPT_FILE" --local-dir "$CE_HOME/ckpt" >>"$log" 2>&1 \
      || { tail -20 "$log"; exit 1; }
  fi
  echo "$CE_CKPT_SHA256  $ckpt" | sha256sum -c - >/dev/null 2>&1 || { echo "checkpoint sha256 mismatch: $ckpt"; rm -f "$ckpt"; exit 1; }
  touch "$ckpt.ok"
fi

"$PY" - <<'EOF'
import torch, vggt_omega, numpy
print("ready: torch %s, cuda %s, %s, numpy %s" % (torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU", numpy.__version__))
EOF
