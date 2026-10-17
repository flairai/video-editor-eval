"""Ask Gemini one question about each audio file in a jobs file and save the answers.

    GEMINI_API_KEY=... python3 build/audit_audio.py build/prompts/stem-audit.json \
        build/intermediate/sam-audio/gemini_audit_flash.json gemini-3.8-flash

The jobs file maps a name to [audio path, question]. gemini-4-argon returned empty or
garbled answers for mostly silent stems; gemini-3.8-flash answered them.
"""

import json
import os
import sys
import time

from google import genai
from google.genai import types


def main(jobs_file: str, out_file: str, model: str) -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    answers = {}
    for name, (path, question) in json.load(open(jobs_file)).items():
        f = client.files.upload(file=path)
        while f.state and f.state.name == "PROCESSING":
            time.sleep(3)
            f = client.files.get(name=f.name)
        response = client.models.generate_content(
            model=model, contents=[types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type)), question],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=24000))
        answers[name] = response.text
        print("=====", name, "\n", response.text, flush=True)
    json.dump(answers, open(out_file, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
