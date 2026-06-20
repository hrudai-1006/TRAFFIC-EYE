# Traffic-Eye AI 🚦
**Flipkart Gridlock Hackathon 2.0 — Round 2 Prototype**

Traffic-Eye AI is an autonomous, multi-module system that processes traffic camera footage to detect, track, and log 7 different types of traffic violations using a combination of trained deep learning models and rule-based heuristics.

## Features
- **7 Violation Modules**:
  - 🧠 *Trained Models*: Helmet Detection, Seatbelt Detection, License Plate Recognition (ANPR).
  - 📏 *Rule-Based*: Triple Riding, Illegal Parking, Wrong-Side Driving, Red-Light Violation.
- **Intelligence Layer**: Calculates Risk Scores, tracks Repeat Offenders using SQLite, and applies Confidence Gating (only ≥85% confidence violations are logged formally).
- **Agentic Routing**: A ReAct-style agent evaluates the severity of formal violations and routes alerts to the appropriate simulated jurisdiction officer.
- **Analytics Dashboard**: A Streamlit dashboard visualizes violation heatmaps, charts, and calculates a custom Traffic Compliance Index (TCI).

## Requirements
- Python 3.11+
- Apple Silicon (M-series) or CPU (CUDA is not natively required/assumed, runs on MPS)

## Setup
```bash
# Clone and enter directory
cd traffic-eye-ai

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run the Pipeline
Run the full detection pipeline on a video file:
```bash
python pipeline.py --source data/test_videos/your_video.mp4 --output output/
```
The output will contain the annotated video. Check the `evidence_store/` directory for generated violation images, JSON records, and the SQLite repeat-offender database.

### 2. View Analytics Dashboard
```bash
streamlit run dashboard/app.py
```

## Documentation
- `DATA_SOURCES.md`: Details of the public datasets used to train the models.
- `LIMITATIONS.md`: Honest accounting of system limitations and simulated components.
