# core/debug_draw_decisions.py

import cv2
import numpy as np
from pathlib import Path

from core.frame_source import open_frames
from core.roi import ROIMap
from core.privacy_decision import PrivacyDecisionEngine
from core.vehicle_detector import VehicleDetector


VIDEO = Path("recording/incoming/tiandy_20260207_162108.mp4")
ROI_JSON = Path("reports/roi/parking_zones.json")


def draw_bbox(img, bbox, color, label):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    roi = ROIMap.from_json(ROI_JSON)
    engine = PrivacyDecisionEngine(roi)
    detector = VehicleDetector(conf=0.25)

    with open_frames(VIDEO, max_frames=1) as src:
        meta = src.metadata
        frame = next(iter(src))

        img = np.frombuffer(
            frame.data, dtype=np.uint8
        ).reshape((meta.height, meta.width, 3))

        # RGB → BGR для OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        boxes = detector.detect(img)

        for i, bbox in enumerate(boxes):
            decision = engine.decide(
                frame_index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                bbox=bbox,
            )

            # цвет bbox машины
            draw_bbox(
                img,
                bbox,
                color=(255, 0, 0),  # синий
                label=f"vehicle #{i}",
            )

            # цвет search ROI
            roi_color = (0, 0, 255) if decision.policy.name == "aggressive" else (0, 255, 255)

            s = decision.plate_search_roi
            draw_bbox(
                img,
                (s.x1, s.y1, s.x2, s.y2),
                roi_color,
                f"{decision.zone or 'flow'} | {decision.policy.name}",
            )

        # окно с изменяемым размером
        cv2.namedWindow("decision_debug", cv2.WINDOW_NORMAL)
        cv2.imshow("decision_debug", img)
        print("Нажми любую клавишу для выхода")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
