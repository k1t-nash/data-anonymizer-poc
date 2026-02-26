"""
Interactive multi-zone ROI drawer for a static camera scene.

Controls:
- Left Click       : add point to active zone
- U / Backspace    : undo last point (active zone)
- R                : reset active zone
- N                : finalize active zone and start new one
- S                : save all zones to JSON
- Q / ESC          : quit without saving

Active zone  : green
Finalized zones : blue

Output: reports/roi/parking_zones.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from core.frame_source import open_frames


WINDOW_NAME = "ROI Drawer | Click: add | N: next zone | S: save | Q: quit"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m core.roi_draw <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print("ERROR: file not found:", video_path)
        sys.exit(1)

    # Load reference frame
    with open_frames(video_path, max_frames=1) as src:
        meta = src.metadata
        frame = next(iter(src))

    img = np.frombuffer(frame.data, dtype=np.uint8) \
        .reshape((meta.height, meta.width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    polygons: list[dict] = []
    active_points: list[tuple[int, int]] = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= x < meta.width and 0 <= y < meta.height:
                active_points.append((x, y))
            else:
                print("Click outside image bounds ignored")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 900)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    try:
        while True:
            canvas = img.copy()

            # Draw finalized zones (blue)
            for poly in polygons:
                pts = np.array(poly["points"], dtype=np.int32)
                cv2.polylines(
                    canvas,
                    [pts],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=2,
                )

            # Draw active zone (green)
            for p in active_points:
                cv2.circle(canvas, p, 5, (0, 255, 0), -1)

            if len(active_points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.array(active_points, dtype=np.int32)],
                    isClosed=False,
                    color=(0, 255, 0),
                    thickness=2,
                )

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF

            # Quit
            if key in (27, ord("q")):
                print("Exit without saving")
                break

            # Undo last point
            if key in (ord("u"), 8):
                if active_points:
                    active_points.pop()
                    print("Undo last point")

            # Reset active zone
            if key == ord("r"):
                active_points.clear()
                print("Active zone reset")

            # Finalize active zone
            if key == ord("n"):
                if len(active_points) < 3:
                    print("Need at least 3 points to finalize zone")
                    continue

                zone_name = f"parking_zone_{len(polygons) + 1}"
                polygons.append(
                    {
                        "name": zone_name,
                        "points": active_points.copy(),
                    }
                )
                active_points.clear()
                print(f"Zone finalized: {zone_name}")

            # Save all zones
            if key == ord("s"):
                if active_points:
                    print("Finalize active zone with N before saving")
                    continue

                if not polygons:
                    print("No zones to save")
                    continue

                out_dir = Path("reports/roi")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "parking_zones.json"

                data = {
                    "image_width": meta.width,
                    "image_height": meta.height,
                    "coordinate_system": "pixel (origin top-left, x right, y down)",
                    "zones": polygons,
                }

                with open(out_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"Saved {len(polygons)} zones → {out_path}")
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
