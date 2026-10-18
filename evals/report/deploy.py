"""Publish a built page on Cloudflare Pages, so anyone can open it at a public URL.

    uv run python -m evals.report.deploy jobs/reports/benchmark --project video-editor-eval --account <id>
    uv run python -m evals.report.deploy jobs/reports/benchmark --project video-editor-eval --dry-run

It copies the page's folder (index.html, manifest.json, media/) to jobs/.deploy/<project>/ with every file
inside Pages' limits: at most 25 MiB a file and 20,000 files. A video over the limit is re-encoded at the
bitrate that fits, cached by content beside the report's previews, and keeps its name, so the page is
unchanged. Beside them go a 404 page and a small worker (pages_worker.js) that answers the byte-range
requests a browser needs to seek in a video, which Pages' own file server ignores. Before anything leaves this machine, every text file is scanned for keys: the value of every
*_KEY, *_TOKEN, *_SECRET or *_PASSWORD variable in the environment or in the --env-file files, and the
shapes keys take (sk-, AIza, ghp_, hf_, private key blocks). One found stops the deploy, naming the file.
Then `wrangler pages deploy` uploads the copy; wrangler (installed, or through npx) must be logged in
(`npx wrangler login`), and --account picks the Cloudflare account when it sees several.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from evals.core.config import JOBS, ROOT, load_env_files, rel, resolve, settings
from evals.core.secrets import secret_values

LIMIT = 25 * 2**20 - 2**18        # Pages' 25 MiB a file, less a margin
MAX_FILES = 20_000
WORKER = Path(__file__).with_name("pages_worker.js")   # byte ranges for the video and audio
STREAMED = ("mp4", "m4v", "mov", "webm", "m4a", "mp3", "wav", "ogg")
NOT_FOUND = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>Not found</title><style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#000;'
             'color:#a1a1aa;font:15px/1.5 system-ui,sans-serif}a{color:#fafafa}</style><p>Not found. <a href="/">Back to the page</a></p>')
TEXT = (".html", ".json", ".txt", ".md", ".csv", ".js", ".css", ".svg")
SHAPES = {
    "an OpenAI key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}",
    "an Anthropic key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "a Google API key": r"\bAIza[0-9A-Za-z_-]{35}",
    "a GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{36,}",
    "a Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "an AWS key id": r"\bAKIA[0-9A-Z]{16}\b",
    "a private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}


def _duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def fit_video(src: Path, cache: Path, limit: int = LIMIT) -> Path:
    """A copy of the video under `limit` bytes, at the highest bitrate that fits (cached by its content)."""
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:20]
    target = cache / ("web-%s-%d.mp4" % (sha, limit))
    if target.exists():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    seconds, audio_kbps = max(_duration(src), 1.0), 128
    share = 0.92
    for _ in range(4):
        video_kbps = max(150, int(limit * 8 / 1000 / seconds * share) - audio_kbps)
        tmp = target.with_suffix(".tmp.mp4")
        subprocess.run(["nice", "ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "medium",
                        "-b:v", "%dk" % video_kbps, "-maxrate", "%dk" % int(video_kbps * 1.2), "-bufsize", "%dk" % (video_kbps * 2),
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "%dk" % audio_kbps, "-movflags", "+faststart", str(tmp)], check=True)
        if tmp.stat().st_size <= limit:
            os.replace(tmp, target)
            return target
        share *= 0.85
        tmp.unlink()
    raise SystemExit("could not fit %s under %d bytes" % (src, limit))


def stage(page: Path, out: Path, cache: Path) -> dict:
    """The page's folder copied to `out` (linked where it can be), every file within Pages' limits."""
    page = page.resolve()
    if out.exists():
        shutil.rmtree(out)
    shrunk, files = [], 0
    for src in sorted(p for p in page.rglob("*") if p.is_file()):
        dst = out / src.relative_to(page)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.stat().st_size > LIMIT:
            if src.suffix.lower() not in (".mp4", ".m4v", ".mov"):
                raise SystemExit("%s is over Pages' 25 MiB and is not a video to re-encode" % rel(src))
            print("  fitting %s (%.0f MB)" % (src.relative_to(page), src.stat().st_size / 1e6), file=sys.stderr)
            src, shrunk = fit_video(src, cache), shrunk + [str(dst.relative_to(out))]
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        files += 1
    if files > MAX_FILES:
        raise SystemExit("%d files: over Pages' %d" % (files, MAX_FILES))
    sizes = {"/" + p.relative_to(out).as_posix(): p.stat().st_size for p in sorted(out.rglob("*"))
             if p.is_file() and p.suffix.lower().lstrip(".") in STREAMED}
    worker = WORKER.read_text()
    assert "const SIZES = {};" in worker
    (out / "_worker.js").write_text(worker.replace("const SIZES = {};", "const SIZES = %s;" % json.dumps(sizes, sort_keys=True), 1))
    (out / "_routes.json").write_text(json.dumps({"version": 1, "include": ["/media/*.%s" % e for e in STREAMED], "exclude": []}))
    (out / "404.html").write_text(NOT_FOUND)  # a missing file is a 404, not the page
    return {"files": files, "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()), "shrunk": shrunk}


def scan(folder: Path) -> list[str]:
    """Every key found in the folder's text files, as "file: what" (never the value itself)."""
    values = secret_values()
    found = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT:
            continue
        text = p.read_text(errors="replace")
        for value, name in values:
            if any(form in text for form in {value, json.dumps(value)[1:-1]}):  # as it is, and as JSON escapes it
                found.append("%s: the value of %s" % (p.relative_to(folder), name))
        for what, pattern in SHAPES.items():
            if re.search(pattern, text):
                found.append("%s: %s" % (p.relative_to(folder), what))
    return found


def wrangler(args: list[str], account: str | None) -> subprocess.CompletedProcess:
    """Cloudflare's CLI: installed, or through npx."""
    cmd = ["wrangler"] if shutil.which("wrangler") else ["npx", "--yes", "wrangler@4"] if shutil.which("npx") else None
    if not cmd:
        raise SystemExit("the deploy needs Cloudflare's wrangler: install Node.js, then `npx wrangler login`")
    env = {**os.environ, **({"CLOUDFLARE_ACCOUNT_ID": account} if account else {})}
    return subprocess.run([*cmd, *args], capture_output=True, text=True, env=env, cwd=ROOT)


def projects(account: str | None) -> dict[str, str]:
    """The account's Pages projects, by name, with each one's pages.dev address."""
    listed = wrangler(["pages", "project", "list", "--json"], account)
    if listed.returncode != 0:
        raise SystemExit("wrangler cannot list Pages projects (log in with `wrangler login`, or pass --account):\n" + listed.stderr[-800:])
    out = {}
    for p in json.loads(listed.stdout[listed.stdout.index("["):]):
        name = p.get("Project Name") or p.get("name")
        domains = p.get("Project Domains") or p.get("domains") or ""
        out[name] = next((d.strip() for d in str(domains).split(",") if d.strip().endswith(".pages.dev")), "%s.pages.dev" % name)
    return out


def deploy(folder: Path, project: str, account: str | None, branch: str = "main") -> str:
    """Upload the folder as the project's production deployment; returns the project's address."""
    if project not in projects(account):
        made = wrangler(["pages", "project", "create", project, "--production-branch", branch], account)
        if made.returncode != 0:
            raise SystemExit("could not create the Pages project %s:\n%s" % (project, made.stderr[-800:]))
    done = wrangler(["pages", "deploy", str(folder), "--project-name", project, "--branch", branch, "--commit-dirty=true"], account)
    print(done.stdout[-1500:], file=sys.stderr)
    if done.returncode != 0:
        raise SystemExit("the deploy failed:\n" + done.stderr[-1500:])
    return "https://" + projects(account)[project].removeprefix("https://")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.report.deploy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page", type=Path, help="a built page's folder (holding index.html), such as jobs/reports/benchmark")
    ap.add_argument("--project", required=True, help="the Cloudflare Pages project; its address is https://<project>.pages.dev")
    ap.add_argument("--account", help="the Cloudflare account id, when wrangler can see more than one")
    ap.add_argument("--env-file", type=Path, action="append", help="KEY=value files whose keys must not appear in the page (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="stage and scan only: upload nothing")
    args = ap.parse_args(argv)
    load_env_files(args.env_file)
    if args.page.name == "index.html":
        args.page = args.page.parent
    if not (args.page / "index.html").is_file():
        raise SystemExit("%s has no index.html: build the page first (uv run evals report ...)" % args.page)
    out = JOBS / ".deploy" / args.project
    stats = stage(args.page, out, resolve(settings()["report"]["cache"]))
    print("staged %s: %d files, %.0f MB, %d videos re-encoded to fit" % (rel(out), stats["files"], stats["bytes"] / 1e6, len(stats["shrunk"])))
    found = scan(out)
    if found:
        raise SystemExit("not deployed: keys in the page\n  " + "\n  ".join(found))
    print("no keys found in %d text files" % sum(1 for p in out.rglob("*") if p.suffix.lower() in TEXT))
    if args.dry_run:
        return
    print("published at", deploy(out, args.project, args.account))


if __name__ == "__main__":
    main()
