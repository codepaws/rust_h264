"""Generate a deterministic MBAFF CABAC regression stream.

Requires FFmpeg with the libx264 encoder. The encoder and decoder are used as
executables only; no codec implementation source is used. Run from repo root.
"""

from pathlib import Path
import hashlib
import subprocess
import tempfile


WIDTH = 64
HEIGHT = 32
FRAMES = 4
STEM = "mbaff_cabac_skipped_top_field_bottom_test"
OUT = Path(__file__).resolve().parent


def source_frame(frame):
    y = bytearray(WIDTH * HEIGHT)
    u = bytearray(WIDTH * HEIGHT // 4)
    v = bytearray(WIDTH * HEIGHT // 4)
    for row in range(HEIGHT):
        for col in range(WIDTH):
            # Constant, distinct fields make a wrong frame/field row mapping
            # immediately visible while keeping intra prediction simple.
            y[row * WIDTH + col] = 200 + frame * 10 if row & 1 else 40
    for row in range(HEIGHT // 2):
        for col in range(WIDTH // 2):
            u[row * WIDTH // 2 + col] = 128
            v[row * WIDTH // 2 + col] = 128
    return y + u + v


def run(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    stream = OUT / f"{STEM}.h264"
    decoded = OUT / f"{STEM}.yuv"
    with tempfile.TemporaryDirectory(dir=OUT) as directory:
        raw = Path(directory) / "source.yuv"
        source = b"".join(source_frame(i) for i in range(FRAMES))
        raw.write_bytes(source)
        run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "yuv420p", "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", "25", "-i", str(raw), "-frames:v", str(FRAMES),
            "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
            "-x264-params", "interlaced=1:tff=1:cabac=1:bframes=0:ref=1:keyint=4:min-keyint=4:scenecut=0:mbtree=0:rc-lookahead=0:threads=1:no-deblock=1:weightp=0",
            "-qp", "22", "-f", "h264", str(stream))
        check = Path(directory) / "independent_decode.yuv"
        run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(stream),
            "-frames:v", str(FRAMES), "-pix_fmt", "yuv420p", "-f", "rawvideo", str(check))
        if check.read_bytes() != source:
            raise RuntimeError("encoded stream is not byte-exact to analytic source")
        decoded.write_bytes(source)
    for path in (stream, decoded):
        print(f"{path.name} {path.stat().st_size} bytes sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
