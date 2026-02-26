import subprocess
import json
import sys
from pathlib import Path


def run(cmd):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def ffprobe_video(path: Path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,time_base",
        "-show_entries",
        "format=duration",
        "-of", "json",
        str(path),
    ]
    rc, out, err = run(cmd)
    if rc != 0:
        raise RuntimeError(err)
    return json.loads(out)


def main():
    if len(sys.argv) != 2:
        print("Usage: python core/ffprobe_info.py <video_file>")
        sys.exit(1)

    video = Path(sys.argv[1])
    if not video.exists():
        print("ERROR: file not found:", video)
        sys.exit(1)

    info = ffprobe_video(video)

    stream = info["streams"][0]
    duration = float(info["format"]["duration"])

    print("Codec:", stream["codec_name"])
    print("Resolution:", f'{stream["width"]}x{stream["height"]}')
    print("avg_frame_rate:", stream["avg_frame_rate"])
    print("r_frame_rate:", stream["r_frame_rate"])
    print("time_base:", stream["time_base"])
    print("Duration (sec):", round(duration, 2))


if __name__ == "__main__":
    main()
