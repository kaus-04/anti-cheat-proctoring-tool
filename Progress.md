# Project Progress Anchor

## 1) Project Status Map
| Feature | Status | Primary File(s) |
|---|---|---|
| Face Presence Detection (MTCNN) | Completed | `src/detection/face_detection.py` |
| Eye Ratio + Gaze Direction Estimation (MediaPipe FaceMesh + Iris Fusion) | In-Progress | `src/detection/eye_tracking.py`, `src/main.py` |
| Gaze-Away Violation Logging Pipeline | In-Progress | `src/main.py` (branch currently commented for `GAZE_AWAY`) |
| Mouth Movement Detection | Completed | `src/detection/mouth_detection.py`, `src/main.py` |
| Multi-Face Detection | Completed | `src/detection/multi_face.py`, `src/main.py` |
| Object Detection (Normal: YOLO26n, Strict: RT-DETR) | Completed | `src/detection/object_detection.py`, `src/main.py`, `config/config.yaml` |
| Real-Time Audio VAD (WebRTCVAD + RMS Debounce) | Completed | `src/main.py` |
| Interview Mode Policy (speech/mouth allowed) | Completed | `src/main.py`, `config/config.yaml` |
| Webcam Recording | Completed | `src/utils/video_utils.py`, `src/main.py` |
| Screen Recording | Completed | `src/utils/screen_capture.py`, `src/main.py` |
| Violation Screenshot Capture | Completed | `src/utils/screenshot_utils.py`, `src/main.py` |
| Violation JSON Persistence | Completed | `src/utils/violation_logger.py` |
| Report Generation (PDF/HTML + Timeline + Heatmap) | Completed | `src/reporting/report_generator.py`, `src/reporting/templates/base_report.html` |
| Dashboard Upload + Background Analysis Jobs | Completed | `src/dashboard/app.py`, `src/dashboard/templates/dashboard.html` |
| Dashboard Report Download + Visualizations | Completed | `src/dashboard/app.py`, `src/dashboard/templates/dashboard.html` |
| Code Plagiarism Analysis API (`/api/code-analysis`) | Completed | `src/dashboard/app.py` |
| AI-Generated Code Probability Scoring | Completed | `src/dashboard/app.py` |
| Dashboard Code Analysis Tab (input + progress bars) | Completed | `src/dashboard/templates/dashboard.html` |
| Identity Verification / Candidate Matching | Planned | N/A |

## 2) Technical Stack & Dependencies
- `opencv-python`: frame capture, drawing overlays, display loop, and video encoding helpers.
- `facenet-pytorch` (`MTCNN`): primary face presence detector in live loop.
- `mediapipe` (`FaceMesh`, `refine_landmarks=True`): eye landmarks, iris-aware gaze estimation, mouth geometry.
- `ultralytics` (`YOLO`, `RTDETR`): dual object detection backends (`YOLO26n` normal mode, `RT-DETR` strict mode).
- `pyaudio`: microphone stream capture (`16kHz`, mono, int16).
- `webrtcvad` (Mode 3): strict speech activity filtering from raw audio frames.
- `audioop`: RMS energy calculation for sustained non-speech loudness events.
- `flask`: dashboard server, upload API, job polling API, artifact/report serving.
- `copydetect`: plagiarism similarity analysis against local `src/reference_solutions/`.
- `transformers`: AI-generated text/code probability scoring via startup-loaded text classifier pipeline.
- `matplotlib`: timeline + heatmap rendering for report artifacts.
- `jinja2`: report HTML templating.
- `pdfkit`: HTML-to-PDF export.
- `gTTS` + `pygame`: audible alert synthesis/playback.
- `pyyaml`: runtime configuration loading.

## 3) Architecture Overview (`src/`)
```text
src/
  main.py                      # Runtime orchestrator (capture loop, detectors, violation routing, mode policy, audio VAD thread)
  detection/
    face_detection.py          # Face presence detection via MTCNN
    eye_tracking.py            # EAR + fused head/iris gaze direction estimation
    mouth_detection.py         # Mouth movement detection
    multi_face.py              # Multiple-face detection
    object_detection.py        # Dual backend object detector (YOLO26n normal / RT-DETR strict)
    audio_detection.py         # Legacy/alternate audio monitor implementation (currently not main execution path)
  utils/
    video_utils.py             # Webcam recording writer lifecycle
    screen_capture.py          # Screen recorder lifecycle
    screenshot_utils.py        # Violation frame capture to disk
    violation_logger.py        # Violation JSON persistence API
    logging.py                 # Alert log persistence helpers
    alert_system.py            # TTS + playback alert delivery
  reporting/
    report_generator.py        # Builds stats + images + PDF/HTML report artifact
    templates/base_report.html # Report template
  dashboard/
    app.py                     # Flask APIs (upload/job/artifacts/stats/download/code-analysis)
    templates/dashboard.html   # Dashboard UI (proctoring tab + code analysis tab)
```

## 4) Current Session Handover
### Most Recent Implemented Logic
- Added **session mode policy** in `main.py`:
  - `exam` = strict proctoring.
  - `interview` = speaking allowed (audio monitor disabled, mouth violations disabled).
- Added CLI mode override: `--mode exam|interview`.
- Added config mode anchor: `session.mode` in `config/config.yaml`.
- Added explicit `Mouth: Allowed` status behavior in interview mode.
- Added real-time audio VAD thread in main runtime path:
  - WebRTCVAD mode 3 speech gate.
  - RMS loudness path for non-speech spikes.
  - Rolling debounce buffers and cooldown-based violation logging.
- Object detection was tuned for phone recall:
  - class-name normalization/aliases.
  - class-specific confidence support.
  - lower confidence floor for `cell phone`.
- Finalized object detection runtime modes:
  - Normal mode uses `YOLO26n` (fast path).
  - Strict mode uses `RT-DETR` (higher precision path).
  - Added CLI flags: `--strict-objects`, `--disable-objects`.
- Added code analysis subsystem in dashboard:
  - `POST /api/code-analysis` accepts raw code.
  - Plagiarism scoring via `copydetect` (with safe fallback scorer).
  - AI probability scoring via startup-loaded `transformers` pipeline.
  - Risk level classification and warning propagation.
- Added dashboard Code Analysis tab:
  - code textarea input.
  - run analysis action.
  - plagiarism/AI progress bars + risk badge.
- Hardened startup behavior:
  - dashboard no longer crashes when `transformers` is missing.
  - AI detector is marked unavailable with warning instead.

### Next Steps / Known Gaps
- Re-enable and harden **`GAZE_AWAY` violation branch** in `main.py` (currently commented).
- Consolidate gaze logic ownership:
  - `eye_tracking.py` has fused gaze logic, but main routing currently does not emit gaze violation events.
- Decide on single audio implementation path:
  - `src/main.py` thread vs `src/detection/audio_detection.py` (legacy duplication).
- Replace mocked dashboard stats (`face_detected/current_activity/probability`) with live runtime feed.
- Add persistent session metadata (`candidate_id`, `mode`) into violations/report header.
- Add tests for mode-policy behavior (exam vs interview) and object threshold regressions.
- Add dashboard toggle passthrough for object mode (`normal`/`strict`) and disable flag.
- Install and validate `copydetect` in runtime environment (currently optional fallback path exists).
- Install and validate `transformers` model download in runtime environment for live AI scoring.

## 5) Configuration State (`config/config.yaml`)
### Session / Runtime
- `session.mode`: `"exam"` (valid: `exam`, `interview`)
- `video.source`: `0`
- `video.resolution`: `[1280, 720]`
- `video.fps`: `30`
- `screen.recording`: `true`

### Detection Thresholds
| Domain | Key | Value |
|---|---|---|
| Face | `detection.face.detection_interval` | `5` |
| Face | `detection.face.min_confidence` | `0.8` |
| Eyes | `detection.eyes.gaze_threshold` | `2` |
| Eyes | `detection.eyes.blink_threshold` | `0.3` |
| Eyes | `detection.eyes.gaze_sensitivity` | `15` |
| Eyes | `detection.eyes.consecutive_frames` | `3` |
| Mouth | `detection.mouth.movement_threshold` | `3` |
| Multi-face | `detection.multi_face.alert_threshold` | `5` |
| Objects | `detection.objects.strict_mode` | `false` |
| Objects | `detection.objects.min_confidence` | `0.35` |
| Objects | `detection.objects.class_min_confidence.book` | `0.35` |
| Objects | `detection.objects.class_min_confidence.cell phone` | `0.25` |
| Objects | `detection.objects.detection_interval` | `1` |
| Objects | `detection.objects.max_fps` | `5` |
| Objects | `detection.objects.imgsz` | `1024` |
| Objects | `detection.objects.model_candidates` | `["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]` |
| Objects | `detection.objects.strict_model_candidates` | `["rtdetr-l.pt", "rtdetr-x.pt"]` |
| Audio | `detection.audio_monitoring.sample_rate` | `16000` |
| Audio | `detection.audio_monitoring.energy_threshold` | `1200` |
| Audio | `detection.audio_monitoring.debounce_window_frames` | `20` |
| Audio | `detection.audio_monitoring.speech_trigger_frames` | `6` |
| Audio | `detection.audio_monitoring.loud_trigger_frames` | `8` |
| Audio | `detection.audio_monitoring.violation_cooldown_seconds` | `4.0` |
| Logging | `logging.alert_cooldown` | `10` |

### Reporting / Outputs
- `global.output_path`: `./reports`
- `reporting.output_dir`: `./reports/generated`
- `reporting.image_dir`: `./reports/generated/images`
- `logging.log_path`: `./logs`
