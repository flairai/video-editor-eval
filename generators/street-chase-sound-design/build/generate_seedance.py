"""Generate one Seedance 2.5 text-to-video clip through fal's queue and download the MP4.

    FAL_KEY=... python3 build/generate_seedance.py reference/seedance-prompt.md \
        sources/graphite-run-v1.mp4 30 16:9 1080p

Sends the prompt file as written, with native audio on, polls the queue every 10 s and
saves the video plus `<out>.job.json` (queue status, result URL and seed). Seedance
samples a new seed on every call, so a rerun makes a different film.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

ENDPOINT = "https://queue.fal.run/bytedance/seedance-2.5/text-to-video"


def main(prompt_file: str, out_file: str, duration: str, ratio: str, resolution: str) -> None:
    headers = {"Authorization": "Key " + os.environ["FAL_KEY"], "Content-Type": "application/json"}
    body = {"prompt": Path(prompt_file).read_text(), "duration": str(duration), "generate_audio": True,
            "resolution": resolution, "aspect_ratio": ratio, "bitrate_mode": "standard"}
    r = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    job, start = r.json(), time.time()
    while True:
        time.sleep(10)
        status = requests.get(job["status_url"], headers=headers, timeout=60).json()
        print("%4ds %s" % (time.time() - start, status.get("status")), flush=True)
        if status.get("status") in ("COMPLETED", "FAILED", "ERROR", "CANCELLED"):
            break
    result = requests.get(job["response_url"], headers=headers, timeout=60)
    out = Path(out_file)
    record = {"status": status, "result": result.json() if result.ok else result.text}
    out.with_suffix(".job.json").write_text(json.dumps(record, indent=1))
    if not (result.ok and "video" in result.json()):
        sys.exit(f"generation failed: {result.status_code} {result.text[:800]}")
    video = requests.get(result.json()["video"]["url"], timeout=600)
    out.write_bytes(video.content)
    print("saved", out, len(video.content), "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:6])
