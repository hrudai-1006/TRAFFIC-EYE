# DATA_SOURCES.md — Traffic-Eye AI

## Status Summary

| Category | Status | Detail |
|----------|--------|--------|
| Base detection (YOLO11n) | ✅ **IN USE** | COCO-pretrained, auto-downloaded, no fine-tuning needed |
| Helmet detection dataset | ⏳ **PROPOSED — NOT YET DOWNLOADED** | Candidates identified, download blocked by credentials |
| Seatbelt detection dataset | ⏳ **PROPOSED — NOT YET DOWNLOADED** | Candidates identified, download blocked by credentials |
| License plate detection dataset | ⏳ **PROPOSED — NOT YET DOWNLOADED** | Candidates identified, download blocked by credentials |
| Demo test footage | 🟡 **SYNTHETIC PLACEHOLDER** | Programmatically generated, not real traffic footage |

**No dataset has been downloaded into `data/helmet/`, `data/seatbelt/`, or `data/plates/`.** These directories exist but are empty. No model fine-tuning has been performed. The violation modules for helmet, seatbelt, and ANPR currently fall back to stub/heuristic classifiers that produce deliberately low-confidence results.

---

## 1. Base Vehicle/Person Detection — ✅ IN USE

| Field | Value |
|-------|-------|
| **Model** | YOLO11n (Nano) |
| **Source** | Ultralytics (auto-downloaded pretrained weights) |
| **Training Data** | Microsoft COCO (330K images, 80 classes) |
| **Classes Used** | person, bicycle, car, motorcycle, bus, truck, traffic light |
| **License** | Apache 2.0 (Ultralytics), CC BY 4.0 (COCO) |
| **Download** | Automatic via `from ultralytics import YOLO; YOLO('yolo11n.pt')` |
| **Status** | Working — used as-is for inference, no additional training needed |

---

## 2. Helmet Detection — ⏳ PROPOSED, NOT YET DOWNLOADED

| Field | Value |
|-------|-------|
| **Candidate Dataset** | Helmet Detection |
| **Candidate Source** | Kaggle: `andrewmvd/helmet-detection` OR Roboflow Universe "motorcycle helmet detection" |
| **Estimated Size** | ~5,000+ images |
| **Classes** | `helmet`, `no-helmet` (some include `motorcycle`, `rider`) |
| **Format** | YOLO TXT (converted/exported) |
| **License** | CC BY 4.0 |
| **Candidate Fallback** | "Motorcycle Helmet Detection" by Data Science 173 (~3,100 images) on Roboflow |
| **Current Blocker** | Kaggle CLI requires `~/.kaggle/kaggle.json` credentials (not present). Roboflow requires API key or manual browser export. |
| **Current State** | `data/helmet/` directory is **empty**. Module uses stub classifier returning low-confidence "no helmet" predictions. |

---

## 3. Seatbelt Detection — ⏳ PROPOSED, NOT YET DOWNLOADED

| Field | Value |
|-------|-------|
| **Candidate Dataset** | Seatbelt Detection |
| **Candidate Source** | Roboflow Universe: "Seatbelt Detection" by Traffic Violations project |
| **Estimated Size** | ~250–544 images (small — will require augmentation) |
| **Classes** | `seatbelt`, `no-seatbelt` (or `person-seatbelt`, `person-noseatbelt`) |
| **Format** | YOLO TXT |
| **License** | CC BY 4.0 |
| **Candidate Fallback** | CLIP zero-shot classification ("person wearing seatbelt" vs "not wearing") |
| **Current Blocker** | Same as helmet — Kaggle/Roboflow credentials required. |
| **Current State** | `data/seatbelt/` directory is **empty**. Module uses stub classifier returning low-confidence "no seatbelt" predictions. |

---

## 4. License Plate Detection — ⏳ PROPOSED, NOT YET DOWNLOADED

| Field | Value |
|-------|-------|
| **Candidate Dataset** | Large License Plate Detection Dataset |
| **Candidate Source** | Kaggle: `fareselmenshawii/large-license-plate-dataset` |
| **Estimated Size** | ~27,900 images (25,500 train / 1,200 valid / 1,200 test) |
| **Classes** | `license-plate` (single class, localization only) |
| **Format** | YOLO TXT (native) |
| **License** | Open / Research use |
| **OCR Engine** | EasyOCR (pretrained, no fine-tuning needed for OCR itself) |
| **Candidate Fallback** | Roboflow "License Plate Detection" (~10,125 images) |
| **Current Blocker** | Same as above. |
| **Current State** | `data/plates/` directory is **empty**. Module uses contour-based stub plate detector (classical CV) that returns placeholder text with low confidence. |

---

## 5. Demo Test Footage — 🟡 SYNTHETIC PLACEHOLDER

| Field | Value |
|-------|-------|
| **File** | `data/test_videos/traffic_demo_1.mp4` |
| **What It Actually Is** | **Synthetic** — 150 frames, 1280×720, 30fps. Programmatically generated colored rectangles (gray road band, red/blue car rectangles, red/green signal circle) on a black background. |
| **Purpose** | Pipeline integration testing only — validates that all modules initialize, the frame loop runs, and output video is written correctly. |
| **What It Is NOT** | Not real traffic footage. Not stock footage from Pexels or any other source. |
| **Real Footage Status** | Not yet sourced. Automated download from Pexels was attempted and failed (HTTP 403). Manual sourcing of free-license traffic video is still needed. |

---

## Next Action Required

To unblock model training and real demo capability, a team member needs to perform these steps:

### Datasets
1. **Helmet data**: Create a free Roboflow account → search "helmet detection motorcycle" on Roboflow Universe → find a dataset with >3000 images and YOLO-format annotations → export in YOLOv8 format → extract into `data/helmet/`.
   - Alternative: Install Kaggle CLI, place credentials at `~/.kaggle/kaggle.json`, run `kaggle datasets download -d andrewmvd/helmet-detection`.
2. **Seatbelt data**: Same Roboflow process → search "seatbelt detection" → export → extract into `data/seatbelt/`.
3. **Plate data**: `kaggle datasets download -d fareselmenshawii/large-license-plate-dataset` → extract into `data/plates/`.
   - Alternative: Roboflow Universe → "License Plate Detection" → export.

### Demo Video
4. **Real traffic footage**: Visit [pexels.com/search/videos/traffic/](https://www.pexels.com/search/videos/traffic/) → download 1-2 clips showing mixed vehicles (cars, motorcycles, pedestrians) → save to `data/test_videos/`.

### After Downloads
5. Run `python3 models/helmet_classifier.py` (or equivalent training script) to fine-tune YOLO11n on the downloaded data.
6. Repeat for seatbelt and plate models.
7. Re-run the pipeline on real footage to validate end-to-end with trained models.

---

## Notes
- No proprietary or paid datasets are required — all candidates are open-access
- The COCO-pretrained YOLO11n base model is NOT re-trained — only used for inference
- No fine-tuning has been performed yet for any module; all classifiers currently use fallback stubs
- EasyOCR is installed and operational but has not been used on real plate images yet
