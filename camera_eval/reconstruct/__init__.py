"""VGGT-Omega as a service: server.py runs on the GPU host (torch, vggt_omega), client.py on this machine.

The wire format is small on purpose so any GPU host can serve it: POST /reconstruct with the nine
frames as base64 PNG in JSON, an npz back (camera-from-world extrinsics in OpenCV axes, intrinsics,
depth and confidence as float16, and the images VGGT-Omega saw as uint8). bootstrap.sh installs the
server's dependencies and the checkpoint on a host; camera_eval.gpu starts it there.
"""
