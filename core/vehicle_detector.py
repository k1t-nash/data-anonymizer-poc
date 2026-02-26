from typing import Tuple
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

BBox = Tuple[int, int, int, int]

MIN_VEHICLE_WIDTH  = 40
MIN_VEHICLE_HEIGHT = 30
BBOX_PAD = 20


class VehicleDetector:
    # COCO: car, motorcycle, bus, truck
    VEHICLE_CLASSES = [2, 3, 5, 7]

    def __init__(
        self,
        model_path: str = "yolov8l.pt",
        conf: float = 0.25,
        pad: int = BBOX_PAD,
    ):
        self.conf = conf
        self.pad  = pad
        self.device = 0 if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.device != "cpu":
            self.model.half()  # FP16 — x1.5 быстрее, качество не теряется

        self.tracker = sv.ByteTrack()

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Принимает BGR кадр.
        Возвращает np.ndarray shape (N, 4) dtype int — (x1, y1, x2, y2).
        Пустой результат → np.empty((0, 4)).
        """
        h, w = frame.shape[:2]

        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=1920,
            classes=self.VEHICLE_CLASSES,
            device=self.device,
            verbose=False,
        )

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            self.tracker.update_with_detections(sv.Detections.empty())
            return np.empty((0, 4), dtype=int)

        detections = sv.Detections.from_ultralytics(r)
        tracked = self.tracker.update_with_detections(detections)

        if len(tracked.xyxy) == 0:
            return np.empty((0, 4), dtype=int)

        boxes = tracked.xyxy.copy().astype(int)

        # Padding — номер на бампере не вылетает за кроп
        boxes[:, 0] = np.clip(boxes[:, 0] - self.pad, 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1] - self.pad, 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2] + self.pad, 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3] + self.pad, 0, h)

        # Фильтр мусора — тени и артефакты детекции
        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths >= MIN_VEHICLE_WIDTH) & (heights >= MIN_VEHICLE_HEIGHT)

        return boxes[valid]
