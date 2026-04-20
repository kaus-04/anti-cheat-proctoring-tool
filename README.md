# AI-Powered Online Exam/Interview Cheating Detection System

<!-- ![System Demo](demo.gif) Add a demo gif later -->

A computer vision system that detects suspicious activities during online exams using webcam footage.

## Features

- **Face Presence Detection**: Identifies when student's face is not visible
- **Eye Movement Tracking**: Detects excessive eye movements (left/right/up/down)
- **Gaze Analysis**: Monitors direction of eye gaze
- **Mouth Movement Detection**: Identifies potential talking or whispering
- **Multi-Face Detection**: Alerts when multiple faces appear in frame
- **Same Candidate Recognition**: Enrolls candidate face embeddings and flags `IDENTITY_MISMATCH` if a different person appears
- **Real-time Alerts**: Flags suspicious activities with timestamps
- **Interview Mode Policy**: `exam` mode runs strict checks, while `interview` mode allows speaking (no voice/mouth speaking violations)
- **Dashboard**: Visual interface showing detection metrics and alerts
- **Dashboard Upload + Background Jobs**: Upload recorded sessions and run server-side analysis in headless mode
- **Object Delection**: Object Detection: Detects prohibited objects (cell phone, book, etc.).
- **Screen Recoding**: Continuously captures examinee's screen activity
- **Real-Time Audio VAD**: Uses WebRTCVAD + RMS debounce to detect sustained speech/loud events with cooldown control
- **Alert Speaker**: Delivers real-time verbal warnings via text-to-speech
- **Report Generation**: Creates detailed visual PDF and HTML reports with violations summary, heatmaps, and activity timeline with non-overlapping point annotations  
- **Dual Object Detection Modes**: Normal mode uses `YOLO26n` for speed, strict mode uses `RT-DETR` for higher precision.
- **Code Integrity Analysis**: Dashboard API computes plagiarism score (`copydetect` + fallback scorer) and AI-generated code probability (`transformers`)


## Technologies Used

- Python 3.8+
- OpenCV (for computer vision)
- MediaPipe (for face mesh and landmark detection)
- FaceNet-PyTorch (MTCNN face detection + InceptionResnetV1 face embeddings)
- Ultralytics (YOLO26n / RT-DETR object detection)
- PyAudio + WebRTCVAD + audioop (real-time voice monitoring)
- gTTS + pygame (voice alerts)
- matplotlib + jinja2 + pdfkit (reporting pipeline)
- copydetect + transformers (code plagiarism + AI probability scoring)
- Flask (for dashboard)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/exam-cheating-detection.git
cd exam-cheating-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (if needed):
```bash
python -c "from facenet_pytorch import MTCNN; MTCNN(keep_all=True)"
```

## Usage

1. Configure the system by editing `config/config.yaml`:
```yaml
video:
  source: 0                   # 0 for default webcam
  resolution: [1280, 720]
  fps: 30
  recording_path: "./recordings"

session:
  mode: "exam"                # exam | interview

screen:
  monitor_index: 0           # 0 for primary monitor
  fps: 15                    # Lower FPS for screen recording
  recording: true            # Enable/disable screen recording


detection:
  face:
    detection_interval: 5     # frames
    min_confidence: 0.8
  eyes:
    gaze_threshold: 2          # seconds
    blink_threshold: 0.3       # EAR threshold for blink detection
    gaze_sensitivity: 15       # pixels threshold for gaze detection
    consecutive_frames: 3      # frames for gaze change detection
  mouth:
    movement_threshold: 3     # consecutive frames
  multi_face:
    alert_threshold: 5        # frames
  identity_verification:
    enrollment_samples: 24    # enrollment baseline samples
    check_interval: 10        # verify every N frames
    distance_threshold: 0.45  # cosine distance mismatch threshold
    mismatch_consecutive: 3   # trigger after N consecutive mismatches
    min_face_confidence: 0.90 # clear single-face requirement
  objects:
    strict_mode: false
    min_confidence: 0.35  # Detection confidence threshold
    detection_interval: 1 # frames between detections
    max_fps: 5            # Maximum detection frames per second
    imgsz: 1024
    model_candidates: ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]
    strict_model_candidates: ["rtdetr-l.pt", "rtdetr-x.pt"]
    target_objects: ["book", "cell phone"]
    class_min_confidence:
      book: 0.35
      cell phone: 0.25
  audio_monitoring:
    enabled: true
    sample_rate: 16000
    energy_threshold: 1200
    zcr_threshold: 0.35
    debounce_window_frames: 20
    speech_trigger_frames: 6
    loud_trigger_frames: 8
    violation_cooldown_seconds: 4.0
    whisper_enabled: false  # Enable only when needed
    whisper_model: "tiny.en"
        
logging:
  log_path: "./logs"
  alert_cooldown: 10          # seconds
  alert_system:
    voice_alerts: true  # Enable/disable voice alerts
    alert_volume: 0.8   # Volume level (0.0 to 1.0)
    cooldown: 10        # Minimum seconds between same alert
```

2.Run the main detection system:
```bash
python src/main.py
```

3. (Optional) Run the dashboard in another terminal:
```bash
python src/dashboard/app.py
```
4. Access the dashboard at `http://localhost:5000`

## System Architecture
```
exam_cheating_detection/
├── config/              # Configuration files
├── models/              # Pretrained models
├── src/                 # Source code
│   ├── detection/       # Detection modules
│   ├── reporting/       # Reporting application
│   ├── utils/           # Utility functions
│   ├── dashboard/       # Web dashboard
│   └── main.py          # Main application
├── logs/                # Session logs
└── recordings/          # Recorded video sessions
```

## Customization
You can adjust detection thresholds in `config/config.yaml`:
```yaml
eyes:
  gaze_threshold: 2      # seconds of gaze deviation to trigger alert
  blink_threshold: 0.3   # eye aspect ratio for blink detection

mouth:
  movement_threshold: 3  # consecutive frames of mouth movement
```

### Analyze a prerecorded file by setting:
in the **config.yaml** file
```
video:
  source: "C:/path/to/interview_recording.mp4"
```

## Command Line Interface (CLI) Usage

The `main.py` script supports command-line arguments for flexible execution, including batch processing and server-side analysis. 

### Basic Usage
To run the detection system on a specific video file or webcam:
```bash
python src/main.py --source <path_or_index>
```

### Headless & Batch Analysis
For dashboard integrations, automated testing, or running on servers without a display, you can use the headless and silent flags. This ensures the script runs entirely in the background without triggering UI windows or audio alerts.

**Example: Running a background job from a dashboard**
```bash
python src/main.py --source uploads/interview_recording.mp4 --headless --disable-audio --no-screen-recording
```

### Available Arguments

| Argument | Description |
| :--- | :--- |
| `--source` | **(Required)** The input source. Use an integer (e.g., `0`) for the default webcam, or provide a string path to a video file (e.g., `video.mp4`). |
| `--mode` | Session mode policy: `exam` (strict proctoring) or `interview` (speech allowed, no voice/mouth speaking violations). |
| `--headless` | Runs the analysis without a Graphical User Interface (GUI). Disables OpenCV video display windows. Required for server-side or background execution. |
| `--disable-audio` | Mutes all system audio alerts (e.g., text-to-speech warnings). Useful for silent batch processing. |
| `--disable-objects` | Disables object detection entirely (best for maximum FPS / object checks not needed). |
| `--strict-objects` | Enables strict object mode using RT-DETR backend. Default mode uses YOLO26n for speed. |
| `--no-screen-recording` | Disables the screen capture functionality. Improves performance when only analyzing pre-recorded video files. |

### Interview Mode Example
For interviews where speaking is expected:
```bash
python src/main.py --mode interview
```
This keeps face/object/multi-face checks active but disables speaking-related violations.

### Strict Object Detection Example
For higher-precision object detection (phone/book), use strict mode:
```bash
python src/main.py --strict-objects
```

### Disable Object Detection Example
If object checks are not required and you want the lowest latency:
```bash
python src/main.py --disable-objects
```
### Code Analysis Module

On the dashboard, click the code analysis tab and paste the code.
- `plagiarism_score` is computed using `copydetect` when available, with normalized similarity fallback.
- `ai_probability` is computed via `transformers` text-classification pipeline.
- `risk_level` is derived from max(plagiarism_score, ai_probability).

For plagiarism comparison, add reference files in `src/reference_solutions`. 

## Start Fresh For New Candidate

Before analyzing a new candidate, you can clear previously generated artifacts (reports, logs, recordings, uploads, violation captures) with:

```bash
python scripts/reset_candidate_session.py --yes
```

Preview what will be deleted without deleting anything:

```bash
python scripts/reset_candidate_session.py --dry-run
```
## Troubleshooting
Problem: Eye detection working, but not perfect

Solution:

    - Ensure good lighting on face
    - Remove glasses if they cause glare
    - Adjust camera position to be face-level

Problem: Book/Phone detection working, but not perfect
    -

## Contributing
Contributions are welcome! Please open an issue or pull request for any improvements.

## License
MIT License - See [LICENSE](LICENSE) for details.
