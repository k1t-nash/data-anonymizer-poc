"""
Extract sample frames (indices 0, 100, 200) from a video as PNG files
with timestamps in the filename. Uses the unified frame source.
"""

import sys
from pathlib import Path

from core.frame_source import open_frames


TARGET_INDICES = {0, 100, 200}


def save_frame(data: bytes, width: int, height: int, out_path: Path) -> None:
    import numpy as np
    from PIL import Image

    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    Image.fromarray(img).save(out_path)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python core/frame_extract_with_ts.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print("ERROR: file not found:", video_path)
        sys.exit(1)

    out_dir = Path("reports/frames_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Request enough frames to include index 200
    max_frames = max(TARGET_INDICES) + 1

    print("Extracting frames with timestamps:")
    with open_frames(video_path, max_frames=max_frames) as src:
        meta = src.metadata
        for frame in src:
            if frame.index in TARGET_INDICES:
                fname = f"frame_{frame.index:05d}_t_{frame.timestamp_sec:.2f}.png"
                out_path = out_dir / fname
                save_frame(frame.data, meta.width, meta.height, out_path)
                print(f"Saved {fname}")

                if frame.index >= max(TARGET_INDICES):
                    break


if __name__ == "__main__":
    main()
