"""Isolate gunshots, tire screeches and music from the chase soundtrack with SAM Audio on fal.

fal hosts SAM Audio (`fal-ai/sam-audio/separate` for a text prompt,
`fal-ai/sam-audio/span-separate` for text plus time spans), so no gated
Hugging Face checkpoint or local GPU is needed. Event classes use our measured
spans, which Meta's paper found gives the largest quality gain; music uses a
text prompt over the whole clip, and its residual is an all-effects stem.

    FAL_KEY=... python3 build/separate_stems.py build/intermediate/qc/soundtrack_44k.wav \
        build/intermediate/stems-fallback-mrx/events.json build/intermediate/sam-audio

`events.json` holds `gunshot_onsets_s` and `tire_screeches` (`start_s`/`end_s`), measured
on the first, MRX-separated effects stem (build/fallback-mrx/isolate_v2.py). Each job saves
`<stem>_target.wav` and `<stem>_residual.wav`, 48 kHz mono.
"""

import json
import sys
from pathlib import Path

import fal_client
import requests


def spans(events: dict) -> dict:
    gunshots = [
        {"start": round(max(0.0, onset - 0.05), 3), "end": round(onset + 0.45, 3), "include": True}
        for onset in events["gunshot_onsets_s"]
    ]
    screeches = [
        {"start": round(item["start_s"] - 0.1, 3), "end": round(item["end_s"] + 0.15, 3), "include": True}
        for item in events["tire_screeches"]
    ]
    return {"gunshot": gunshots, "tire screech": screeches}


def main(mix: str, events_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio_url = fal_client.upload_file(mix)
    jobs = {
        name.replace(" ", "_"): (
            "fal-ai/sam-audio/span-separate",
            {"audio_url": audio_url, "prompt": name, "spans": event_spans,
             "reranking_candidates": 4, "acceleration": "quality"},
        )
        for name, event_spans in spans(json.load(open(events_path))).items()
    }
    jobs["music"] = (
        "fal-ai/sam-audio/separate",
        {"audio_url": audio_url, "prompt": "music", "predict_spans": False,
         "reranking_candidates": 2, "acceleration": "quality"},
    )
    for stem, (endpoint, arguments) in jobs.items():
        result = fal_client.subscribe(endpoint, arguments=arguments)
        for part in ("target", "residual"):
            data = requests.get(result[part]["url"], timeout=300).content
            (out / f"{stem}_{part}.wav").write_bytes(data)
        print(f"{stem}: saved target and residual", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
