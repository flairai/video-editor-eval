"""Find where each montage segment sits inside its original broadcast clip.

Audio: GCC-PHAT between the montage's audio for the segment and the original's
audio gives the audio offset. Video: frame matching around that offset gives the
video offset and how much of the segment the original actually covers. The two
offsets must agree (both uploads in sync) before a play is used.

  python3 -E -P build/tools/align.py   (run from the case folder)
Writes build/intermediate/alignment.json and build/intermediate/alignment-qa.jpg.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from media import frame_features, gcc_phat, load_audio, video_info  # noqa: E402

CASE = Path.cwd()
BUILD = CASE / "build"
SR = 16000


def downloads(youtube_id):
    folder = CASE / "sources" / "youtube"
    video = next(p for p in folder.glob(youtube_id + ".*.mp4"))
    audio = next(p for p in folder.glob(youtube_id + ".*.m4a"))
    return video, audio


def main():
    spec = json.loads((BUILD / "plays.json").read_text())
    montage_video = BUILD / spec["montage"]["video"]
    montage_audio, _ = load_audio(BUILD / spec["montage"]["audio"], SR)
    results, qa = [], []
    for play in spec["plays"]:
        ms, me = play["montage_start_s"], play["montage_end_s"]
        video, audio = downloads(play["youtube_id"])
        original_audio, _ = load_audio(audio, SR)
        segment = montage_audio[int(ms * SR): int(me * SR)]
        lag, peak, ratio = gcc_phat(original_audio, segment, SR, max_shift_s=len(original_audio) / SR)
        delta_a = lag - ms  # original_time = montage_time + delta_a

        m_times, m_feats = frame_features(montage_video, ms, me, every=2)
        o_times, o_feats = frame_features(video)
        best = None
        for d in np.arange(delta_a - 0.3, delta_a + 0.3 + 1e-9, 1 / 120):
            idx = np.clip(np.searchsorted(o_times, m_times + d - 0.5 / 120), 0, len(o_times) - 1)
            sims = np.einsum("ij,ij->i", m_feats, o_feats[idx])
            sims[(m_times + d < 0) | (m_times + d > o_times[-1])] = -1
            score = float(np.mean(np.sort(sims)[int(0.2 * len(sims)):]))
            if best is None or score > best[0]:
                best = (score, d, sims)
        score, delta_v, sims = best
        matched = sims >= 0.85
        runs, start = [], None
        for i, ok in enumerate(matched):
            if ok and start is None:
                start = i
            if (not ok or i == len(matched) - 1) and start is not None:
                end = i if ok else i - 1
                runs.append((start, end))
                start = None
        longest = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, -1)
        info = video_info(video)
        result = {
            "key": play["key"], "youtube_id": play["youtube_id"], "original_video": video.name,
            "original_audio": audio.name, "original_fps": info["fps"], "original_size": [info["width"], info["height"]],
            "original_duration_s": round(info["duration_s"], 3),
            "montage_start_s": ms, "montage_end_s": me,
            "audio_offset_s": round(delta_a, 4), "audio_peak": round(peak, 4), "audio_peak_ratio": round(ratio, 2),
            "video_offset_s": round(delta_v, 4), "video_mean_similarity": round(score, 3),
            "av_skew_ms": round((delta_v - delta_a) * 1000, 1),
            "coverage": round(float(np.mean(matched)), 3),
            "matched_montage_range_s": [round(float(m_times[longest[0]]), 3), round(float(m_times[longest[1]]) + 2 / 60, 3)],
        }
        results.append(result)
        print("%-9s audio %+8.3fs (peak %.3f, x%.1f)  video %+8.3fs (sim %.3f)  skew %+6.1f ms  coverage %.0f%%  matched %s"
              % (play["key"], delta_a, peak, ratio, delta_v, score, result["av_skew_ms"], 100 * result["coverage"],
                 result["matched_montage_range_s"]))
        for frac in (0.15, 0.5, 0.85):
            t = ms + frac * (me - ms)
            qa.append((play["key"], t, montage_video, t, video, t + delta_v))
    (BUILD / "intermediate").mkdir(exist_ok=True)
    (BUILD / "intermediate" / "alignment.json").write_text(json.dumps(results, indent=1))

    import av
    tiles = []
    for key, mt, mpath, _, opath, ot in qa:
        pair = []
        for path, t in ((mpath, mt), (opath, ot)):
            c = av.open(str(path)); s = c.streams.video[0]
            c.seek(int(max(0, t - 1) / s.time_base), stream=s, backward=True)
            img = None
            for fr in c.decode(s):
                if float(fr.time) >= t - 0.5 / 60:
                    img = fr.to_image().resize((320, 180)); break
            c.close()
            pair.append(img or Image.new("RGB", (320, 180)))
        tile = Image.new("RGB", (640, 196), "black")
        tile.paste(pair[0], (0, 16)); tile.paste(pair[1], (320, 16))
        ImageDraw.Draw(tile).text((4, 2), "%s  montage %.2fs | original %.2fs" % (key, mt, ot), fill=(255, 255, 0))
        tiles.append(tile)
    sheet = Image.new("RGB", (640 * 3, 196 * ((len(tiles) + 2) // 3)), "black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 3) * 640, (i // 3) * 196))
    sheet.save(BUILD / "intermediate" / "alignment-qa.jpg", quality=85)


if __name__ == "__main__":
    main()
