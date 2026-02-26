# Video Anonymization Pipeline 🎥🔒

A production-grade computer vision pipeline for automated anonymization of CCTV footage.  
Built to solve a real business problem: processing large volumes of surveillance video while ensuring GDPR-compliant privacy protection.

---

## What it does

Processes real 4K CCTV video streams and automatically detects and anonymizes:
- **Faces** — YOLOv8-based face detection with ByteTrack temporal tracking
- **License plates** — SAHI-sliced detection + EasyOCR confirmation to eliminate false positives

Both are replaced with **pixelation** (not blur) — an irreversible method that cannot be reconstructed even via deconvolution after 4K upscale.

---

## Key technical decisions

- **SAHI slicing** for plate detection on 4K frames — standard YOLO misses small objects at this resolution
- **OCR confirmation layer** — a detected region is only anonymized if EasyOCR confirms it contains plate-like text. Eliminates false positives on wheels, bumpers, road signs
- **ByteTrack** for temporal consistency — once a face/plate is confirmed, it stays anonymized across frames without re-running detection every frame
- **FP16 inference** on GPU — significant speed improvement on CUDA-capable hardware
- **FFmpeg pipeline** — frame processing is piped directly into ffmpeg encoder, no intermediate files on disk
- **Pixelation via numpy repeat** — no cv2 dependency for the anonymization step itself, faster and more portable

---

## Stack

| Component | Tool |
|---|---|
| Detection | YOLOv8 (ultralytics) |
| Plate detection | SAHI + custom YOLOv8 |
| OCR confirmation | EasyOCR |
| Tracking | supervision (ByteTrack) |
| Video I/O | FFmpeg + OpenCV |
| Hardware | CUDA (FP16), CPU fallback |
| Runtime | Python 3.10, Ubuntu 24.04 |

---

## Real-world results

Processed **335 real CCTV videos** (4K @ 20fps) from a live parking surveillance system in a single overnight batch run.  
Each video produced an anonymized output package with metadata — ready for dataset use or delivery.

---

## Project structure

```
core/
  batch_anonymize_videos.py   # Main pipeline orchestrator
  face_detector.py            # YOLOv8 face detection + tracking
  plate_detector.py           # SAHI + OCR plate detection + tracking
  vehicle_detector.py         # Vehicle detection (ROI filtering)
  privacy_decision.py         # Decision logic: what gets anonymized and why
  frame_reader.py             # Frame extraction from video
  frame_source.py             # Video source abstraction
  video_probe.py              # FFprobe metadata extraction
  roi.py / roi_draw.py        # Region of interest tools
  policy.py                   # Anonymization policy config

scripts/
  father_plate_anonymizer.py  # Standalone plate anonymizer script
```

---

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/video-anonymization-pipeline
cd video-anonymization-pipeline
pip install -r requirements.txt

# Place input videos in recording/incoming/
python -m core.batch_anonymize_videos
# Anonymized output → dataset_v1/production_packages/
```

---

## Requirements

```
ultralytics
supervision
sahi
easyocr
opencv-python
numpy
torch
```

GPU strongly recommended for 4K processing. CPU fallback is available but slow.

---

## Status

**Proof of Concept — functional and tested on real data.**  
Core anonymization pipeline works end-to-end. Not production-hardened for enterprise deployment without further security audit.

---

## About

Built as a practical learning project — real cameras, real constraints, no toy examples.  
Developed with AI-assisted coding (Claude) as part of exploring applied Computer Vision for privacy and compliance use cases.

*If you're working on something in the CV / video analytics / privacy space — feel free to reach out.*
