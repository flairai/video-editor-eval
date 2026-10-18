// Byte ranges for a published page's video and audio: evals/report/deploy.py uploads this file as _worker.js,
// with SIZES filled in, and _routes.json sends only those files through it. Pages' asset server answers every
// request with the whole file, and a browser cannot seek in media served that way (Safari will not play it at
// all). This answers a Range request with just the bytes asked for. The asset's size is not among the headers a
// worker sees, so it comes from SIZES, the size of every such file in the upload, by path.
const SIZES = {};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const size = SIZES[url.pathname];
    if (request.method !== "GET" || !size) return env.ASSETS.fetch(request);
    const whole = await env.ASSETS.fetch(new Request(url, {headers: {"Accept-Encoding": "identity"}}));
    const headers = new Headers(whole.headers);
    headers.set("Accept-Ranges", "bytes");
    const m = /^bytes=(\d*)-(\d*)$/.exec((request.headers.get("Range") || "").trim());
    if (whole.status !== 200 || !m || (m[1] === "" && m[2] === "")) return new Response(whole.body, {status: whole.status, headers});
    const start = m[1] === "" ? Math.max(0, size - Number(m[2])) : Number(m[1]);
    const end = m[1] === "" || m[2] === "" ? size - 1 : Math.min(Number(m[2]), size - 1);
    if (start > end) {
      await whole.body.cancel();
      return new Response(null, {status: 416, headers: {"Content-Range": `bytes */${size}`}});
    }
    headers.set("Content-Range", `bytes ${start}-${end}/${size}`);
    headers.set("Content-Length", String(end - start + 1));
    const out = new FixedLengthStream(end - start + 1);
    let at = 0;
    const body = start === 0 && end === size - 1 ? whole.body : whole.body.pipeThrough(new TransformStream({
      transform(chunk, c) {
        const from = at;
        at += chunk.byteLength;
        if (at <= start || from > end) return;
        c.enqueue(chunk.subarray(Math.max(start - from, 0), Math.min(end + 1 - from, chunk.byteLength)));
        if (at > end) c.terminate();
      },
    }));
    body.pipeTo(out.writable).catch(() => {});
    return new Response(out.readable, {status: 206, headers});
  },
};
