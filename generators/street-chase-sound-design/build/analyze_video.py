"""Ask Gemini about a video, sampling it at a chosen frame rate, and save the answer.

    GEMINI_API_KEY=... python3 build/analyze_video.py <video> <prompt.md> <answer.md> <model> <fps>

Used for the analysis of the reference chase (build/prompts/infinite-analysis.md, 4 fps)
and the audit of the generated scene (build/prompts/generation-audit.md, 12 fps), both with
gemini-4-argon.
"""

import os
import sys
import time

from google import genai
from google.genai import types


def main(video: str, prompt_file: str, out_file: str, model: str, fps: str) -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    f = client.files.upload(file=video)
    while f.state and f.state.name == "PROCESSING":
        time.sleep(5)
        f = client.files.get(name=f.name)
    video_part = types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                            video_metadata=types.VideoMetadata(fps=float(fps)))
    response = client.models.generate_content(
        model=model, contents=[video_part, open(prompt_file).read()],
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=60000))
    open(out_file, "w").write(response.text or "")
    print("wrote", out_file, len(response.text or ""), "chars")


if __name__ == "__main__":
    main(*sys.argv[1:6])
