import json
import time
from pathlib import Path
from typing import List, Dict, Any


def save_anonymization_metadata(
    video_name: str,
    detections: List[Dict[str, Any]],
    video_meta: Dict[str, Any] = None,
) -> Path:
    """
    Сохраняет метадату анонимизации в JSON.

    detections: список словарей вида:
        {'frame': 0, 'type': 'plate'|'face', 'bbox': [x1,y1,x2,y2]}

    video_meta (опционально):
        {'width': 3840, 'height': 2160, 'fps': 25.0, 'duration_sec': 120.5}

    Возвращает путь к сохранённому файлу.
    """
    output_path = Path("dataset_v1/metadata") / f"{video_name}_meta.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Статистика по типам объектов
    stats: Dict[str, int] = {}
    for d in detections:
        t = d.get("type", "unknown")
        stats[t] = stats.get(t, 0) + 1

    data = {
        "project": "ASB Traffic Data",
        "schema_version": "1.1",
        "video_source": video_name,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anonymization": {
            "level": "high",
            "method": "pixelate",
            "plate_detector": "YOLOv8 + SAHI + ByteTrack + EasyOCR",
            "face_detector": "YOLOv8-face + ByteTrack",
        },
        "statistics": {
            "total_detections": len(detections),
            "by_type": stats,
        },
        "video_metadata": video_meta or {},
        "objects_found": detections,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved: {output_path} ({len(detections)} objects)")
    return output_path
