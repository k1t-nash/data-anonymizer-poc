from typing import List, Tuple
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

BBox = Tuple[int, int, int, int]

# Минимальный размер лица в пикселях — меньше это шум
MIN_FACE_SIZE = 10
# Отступ вокруг bbox лица — лицо не обрезается по краям
FACE_PAD = 10


class FaceDetector:
    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        pad: int = FACE_PAD,
    ):
        self.conf   = conf
        self.pad    = pad
        self.device = 0 if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.device != "cpu":
            self.model.half()  # FP16

        self.tracker = sv.ByteTrack()

    def detect_in_frame(self, image: np.ndarray) -> List[BBox]:
        """
        Принимает BGR кадр.
        Возвращает список (x1, y1, x2, y2) для анонимизации.

        imgsz=1920 обязателен для 4K — лица маленькие, нужно разрешение.
        Трекинг удерживает блюр когда лицо временно перекрыто (столб, другой объект).
        """
        h, w = image.shape[:2]

        results = self.model.predict(
            image,
            conf=self.conf,
            imgsz=1920,
            device=self.device,
            verbose=False,
        )

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []

        detections = sv.Detections.from_ultralytics(r)
        tracked = self.tracker.update_with_detections(detections)

        if len(tracked.xyxy) == 0:
            return []

        final: List[BBox] = []

        for xyxy in tracked.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)

            # Padding с защитой границ
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(w, x2 + self.pad)
            y2 = min(h, y2 + self.pad)

            # Фильтр слишком мелких детекций
            if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                final.append((x1, y1, x2, y2))

        return final
