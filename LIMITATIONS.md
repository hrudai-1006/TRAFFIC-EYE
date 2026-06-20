# LIMITATIONS.md — Traffic-Eye AI

## Current Honest Status (Updated)

The Traffic-Eye AI pipeline is architecturally complete: all 7 violation modules, the intelligence layer, the alert router, and the Streamlit dashboard are coded, compile, and integrate end-to-end. **However, the three trained-model violation modules (helmet, seatbelt, license plate) currently run on rule-based stub/fallback classifiers because no training data has been downloaded and no fine-tuning has been performed.** The base vehicle/person detection (YOLO11n, COCO-pretrained) works correctly on real footage. The demo test video currently in the repo (`traffic_demo_1.mp4`) is a synthetic placeholder (programmatically generated colored shapes), not real or stock traffic footage. The agentic alert routing is simulated (console/file output only, no real SMS/WhatsApp dispatch). See `DATA_SOURCES.md` for the specific blockers on each pending dataset.

---

## 1. Hardware & Performance Constraints
- **Target Hardware**: Developed and tested on Apple Silicon (M4). The code explicitly forces the MPS backend with CPU fallback. It does **not** assume CUDA availability.
- **FPS Expectations**: The target FPS is 8-15 FPS for the *full pipeline* (Detection + Tracking + 7 Violation Modules + OCR + Evidence Generation). This is not 30+ FPS real-time. For a real production deployment, server-side GPUs (e.g., NVIDIA T4/A10G) would be required to maintain 30 FPS across multiple camera feeds.

## 2. Simulated Components
- **Agentic Alert Routing**: The ReAct-style agent decision loop currently categorizes violations by severity and selects the correct officer/zone. However, the final dispatch step is **simulated** (printed to the console and logged to a JSON file). There is no real WhatsApp Business API or Twilio integration due to time and credential constraints.
- **GPS Coordinates**: The system does not use live GPS data. Instead, it relies on a static `camera_locations.json` mapping, tying a fixed `camera_id` to a string location name.

## 3. Data Limitations
- **No Training Data Downloaded**: The `data/helmet/`, `data/seatbelt/`, and `data/plates/` directories are **empty**. No datasets have been downloaded and no model fine-tuning has been performed. The helmet, seatbelt, and ANPR (license plate) violation modules currently run on stub/fallback classifiers — the helmet and seatbelt stubs return deliberately low-confidence "no helmet"/"no seatbelt" predictions, and the ANPR stub uses classical contour-based plate localization. **This is the single highest-priority remaining gap** before the prototype demonstrates real CV classification rather than fallback behavior.
- **Demo Footage Is Synthetic**: The current demo video (`data/test_videos/traffic_demo_1.mp4`) is **not** real traffic footage and is **not** stock footage from Pexels or any other source. It is a programmatically generated 150-frame video (1280×720, 30fps) containing colored rectangles simulating a road, vehicles, and traffic signal on a black background. Its sole purpose is validating that the pipeline plumbing runs end-to-end without errors. Real or stock traffic footage has not yet been sourced.
- **No Bengaluru Dataset**: We do not have access to real Bengaluru traffic CCTV footage. Even once training datasets are downloaded, they will be generic public datasets, not geographically specific to Bengaluru.

## 4. Algorithmic Simplifications
- **Red Light Detection**: We use a static Region of Interest (ROI) and basic HSV color thresholding to determine the signal state (Red/Yellow/Green), rather than a complex CNN classifier. This requires manual configuration of the signal ROI for each camera angle in `config/camera_locations.json`.
- **Wrong-Side Driving**: This module uses a simplified 2D direction vector heuristic based on bounding box centroids. It does not perform full 3D camera calibration or perspective transformation.

## 5. Next Steps for Production
To deploy this system in a real-world scenario:
1. **Download training datasets** and fine-tune the helmet, seatbelt, and plate detection models (see `DATA_SOURCES.md` for specific instructions).
2. **Source real demo footage** — either from Pexels/Pixabay (free stock) or from a real traffic camera feed.
3. Replace simulated alert dispatch with a real SMS/WhatsApp gateway.
4. Upgrade from YOLO11n (Nano) to YOLO11m (Medium) running on cloud GPUs for higher accuracy.
5. Replace the static camera config with an API pulling from the city's actual camera inventory.
6. Train the models on true local CCTV data to handle specific local angles and vehicle types.
