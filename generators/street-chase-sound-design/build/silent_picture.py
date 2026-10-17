"""Copy a video's picture without its sound; the video packets are copied unchanged.

    python3 build/silent_picture.py sources/graphite-run-v1.mp4 $T/environment/workspace/graphite-run.mp4
"""

import sys

import av


def main(src: str, dst: str) -> None:
    with av.open(src) as i, av.open(dst, "w", format="mp4") as o:
        vin = i.streams.video[0]
        vout = o.add_stream_from_template(vin)
        vout.codec_tag = vin.codec_tag
        for packet in i.demux(vin):
            if packet.dts is None:
                continue
            packet.stream = vout
            o.mux(packet)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
