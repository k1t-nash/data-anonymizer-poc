#!/usr/bin/env python3
"""
Video probe utilities.
Проверка ffmpeg и получение метаданных видео через ffprobe.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import NamedTuple


class VideoMeta(NamedTuple):
    width:       int
    height:      int
    fps:         float
    duration_sec: float
    codec:       str


def _run(cmd: list) -> tuple:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def get_video_meta(path: Path) -> VideoMeta:
    """
    Получает метаданные видео через ffprobe.
    Точнее чем cv2.VideoCapture для определения fps и длительности.
    Бросает RuntimeError если ffprobe недоступен или файл повреждён.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Видеофайл не найден: {path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]

    rc, out, err = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe ошибка для {path.name}: {err}")

    try:
        info   = json.loads(out)
        stream = info["streams"][0]
        fmt    = info["format"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raise RuntimeError(f"Не удалось распарсить ffprobe output: {e}")

    # avg_frame_rate приходит как "20/1" или "30000/1001"
    rate = stream.get("avg_frame_rate", "0/1")
    try:
        num, den = map(int, rate.split("/"))
        fps = num / den if den else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return VideoMeta(
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps,
        duration_sec=float(fmt.get("duration", 0)),
        codec=stream.get("codec_name", "unknown"),
    )


def check_ffmpeg() -> bool:
    """Проверяет доступность ffmpeg."""
    rc, _, _ = _run(["ffmpeg", "-version"])
    return rc == 0


def main():
    print("Python:", sys.version.split()[0])

    if not check_ffmpeg():
        print("ERROR: ffmpeg недоступен")
        sys.exit(1)

    print("ffmpeg OK")

    if len(sys.argv) == 2:
        path = Path(sys.argv[1])
        try:
            meta = get_video_meta(path)
            print(f"Codec:      {meta.codec}")
            print(f"Resolution: {meta.width}x{meta.height}")
            print(f"FPS:        {meta.fps:.3f}")
            print(f"Duration:   {meta.duration_sec:.2f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        sample_dir = Path("recording")
        print("Recording dir exists:", sample_dir.exists())


if __name__ == "__main__":
    main()
