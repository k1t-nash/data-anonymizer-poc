"""
Unified frame source: single entry point for opening a video file and
iterating over frames with timestamps.

Responsibilities:
- probe video metadata (ffprobe)
- stream raw frames via ffmpeg
- attach index + timestamp to each frame
- own subprocess lifecycle (no broken pipes)
"""

import json
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional, Iterator


# =============================================================================
# Public types
# =============================================================================


class Frame(NamedTuple):
    index: int
    timestamp_sec: float
    data: bytes


class VideoMetadata(NamedTuple):
    width: int
    height: int
    fps: float
    duration_sec: float
    codec: str


# =============================================================================
# Probe
# =============================================================================


def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _probe(path: Path) -> VideoMetadata:
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
        raise RuntimeError(err or "ffprobe failed")

    info = json.loads(out)
    stream = info["streams"][0]

    rate = stream["avg_frame_rate"]  # e.g. "20/1"
    num, den = map(int, rate.split("/"))
    fps = num / den if den else 0.0

    return VideoMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration_sec=float(info["format"]["duration"]),
        codec=stream["codec_name"],
    )


# =============================================================================
# Frame source
# =============================================================================


class FrameSource(Iterator[Frame]):
    def __init__(
        self,
        proc: subprocess.Popen,
        metadata: VideoMetadata,
        frame_size: int,
        max_frames: Optional[int],
    ) -> None:
        self._proc = proc
        self._meta = metadata
        self._frame_size = frame_size
        self._max_frames = max_frames

        self._index = 0
        self._closed = False

    # ---------------------------------------------------------------------

    @property
    def metadata(self) -> VideoMetadata:
        return self._meta

    # ---------------------------------------------------------------------

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    # ---------------------------------------------------------------------

    def __iter__(self) -> "FrameSource":
        return self

    def __next__(self) -> Frame:
        if self._closed:
            raise StopIteration

        if self._max_frames is not None and self._index >= self._max_frames:
            self.close()
            raise StopIteration

        raw = self._proc.stdout.read(self._frame_size)
        if len(raw) < self._frame_size:
            self.close()
            raise StopIteration

        ts = self._index / self._meta.fps if self._meta.fps else 0.0
        frame = Frame(index=self._index, timestamp_sec=ts, data=raw)
        self._index += 1
        return frame

    # ---------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        try:
            if self._proc.stdout:
                self._proc.stdout.close()
            if self._proc.stderr:
                self._proc.stderr.close()
        finally:
            self._proc.wait()

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


# =============================================================================
# Public API
# =============================================================================


def open_frames(
    video_path: str | Path,
    *,
    max_frames: Optional[int] = None,
    start_time_sec: Optional[float] = None,
    pixel_format: str = "rgb24",
) -> FrameSource:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video file not found: {path}")

    if pixel_format not in ("rgb24", "bgr24"):
        raise ValueError("Only rgb24 / bgr24 supported")

    meta = _probe(path)
    bytes_per_pixel = 3
    frame_size = meta.width * meta.height * bytes_per_pixel

    cmd = ["ffmpeg", "-loglevel", "error"]

    if start_time_sec is not None:
        cmd.extend(["-ss", str(start_time_sec)])

    cmd.extend([
        "-i", str(path),
        "-f", "rawvideo",
        "-pix_fmt", pixel_format,
    ])

    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])

    cmd.append("-")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return FrameSource(
        proc=proc,
        metadata=meta,
        frame_size=frame_size,
        max_frames=max_frames,
    )
