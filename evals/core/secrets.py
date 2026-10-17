"""Keys never leave in anything evals prints or writes.

Keys reach evals only through the environment, set in the shell or read from a gitignored .env
(evals.core.config.load_env_files). The value of every variable whose name ends in _KEY, _TOKEN, _SECRET or
_PASSWORD, 8 characters or longer, is replaced by its name in brackets ([OPENAI_API_KEY]):

    redact(text)        in any text
    dumps(obj, ...)     json.dumps, redacted: what every run.json, records.jsonl, judge.json, trajectory and
                        report page is written with
    install()           wraps sys.stdout and sys.stderr, so prints, errors and tracebacks are redacted too;
                        the evals command calls it first

A subprocess writes to the terminal itself, past these wrappers; none of the ones evals starts is given a
key on its command line.
"""

from __future__ import annotations

import io
import json
import os
import sys

SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
MIN_LENGTH = 8


def secret_values() -> list[tuple[str, str]]:
    """(value, variable name) for every key in the environment, longest value first."""
    found: dict[str, str] = {}
    for name, value in os.environ.items():
        if name.upper().endswith(SUFFIXES) and len(value) >= MIN_LENGTH:
            found.setdefault(value, name)
    return sorted(found.items(), key=lambda kv: -len(kv[0]))


def redact(text: str) -> str:
    for value, name in secret_values():
        for form in {value, json.dumps(value)[1:-1]}:  # as it is, and as JSON escapes it
            if form in text:
                text = text.replace(form, "[%s]" % name)
    return text


def dumps(obj, **kwargs) -> str:
    return redact(json.dumps(obj, **kwargs))


class RedactingStream(io.TextIOBase):
    """A text stream that redacts what is written to it and passes everything else to the stream it wraps."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text: str) -> int:
        self._stream.write(redact(text))
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()

    @property
    def encoding(self):
        return self._stream.encoding

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install() -> None:
    """Redact everything this process prints from now on."""
    if not isinstance(sys.stdout, RedactingStream):
        sys.stdout = RedactingStream(sys.stdout)
    if not isinstance(sys.stderr, RedactingStream):
        sys.stderr = RedactingStream(sys.stderr)
