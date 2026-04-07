import cv2
import yaml
import argparse
import os
import json
import re
import audioop
import pyaudio # type: ignore
import webrtcvad # type: ignore
import threading
from collections import deque
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CACHE_DIR = PROJECT_ROOT / ".runtime-cache"
MPL_CONFIG_DIR = RUNTIME_CACHE_DIR / "matplotlib"
YOLO_CONFIG_DIR = RUNTIME_CACHE_DIR / "ultralytics"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

from detection.face_detection import FaceDetector
from detection.eye_tracking import EyeTracker
from detection.mouth_detection import MouthMonitor
from detection.object_detection import ObjectDetector
from detection.multi_face import MultiFaceDetector
from detection.identity_verification import IdentityVerifier
from utils.video_utils import VideoRecorder
from utils.screen_capture import ScreenRecorder
from utils.logging import AlertLogger
from utils.alert_system import AlertSystem
from utils.violation_logger import ViolationLogger
from utils.screenshot_utils import ViolationCapturer
from reporting.report_generator import ReportGenerator


def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)


def _parse_video_source(source_value):
    if source_value is None:
        return None
    if isinstance(source_value, int):
        return source_value
    source_text = str(source_value).strip()
    return int(source_text) if source_text.isdigit() else source_text


def parse_args():
    parser = argparse.ArgumentParser(description="Exam cheating detection")
    parser.add_argument("--source", help="Video source index (e.g. 0) or file path")
    parser.add_argument(
        "--mode",
        choices=["exam", "interview"],
        help="Session mode: exam (strict) or interview (speech allowed)"
    )
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--disable-audio", action="store_true", help="Disable audio monitoring")
    parser.add_argument("--disable-objects", action="store_true", help="Disable object detector")
    parser.add_argument(
        "--strict-objects",
        action="store_true",
        help="Use strict object detection mode (RT-DETR backend)"
    )
    parser.add_argument(
        "--no-screen-recording",
        action="store_true",
        help="Disable screen recording for this run"
    )
    parser.add_argument("--candidate-id", help="Candidate identifier for report metadata")
    parser.add_argument("--candidate-name", help="Candidate full name for report metadata")
    parser.add_argument("--exam-name", help="Exam/session name for report metadata")
    parser.add_argument("--course-name", help="Course name for report metadata")
    return parser.parse_args()


def _resolve_project_path(path_text):
    project_root = Path(__file__).resolve().parent.parent
    path_obj = Path(path_text)
    return str(path_obj if path_obj.is_absolute() else (project_root / path_obj))


def _safe_int_threshold(value, default_value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default_value
    if numeric <= 1.0:
        return int(numeric * 32768)
    return int(numeric)


def _sanitize_identifier(raw_value, fallback):
    text = str(raw_value or "").strip()
    if not text:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return safe or fallback


def _safe_text(raw_value, fallback):
    text = str(raw_value or "").strip()
    return text if text else fallback


def _build_student_info(config, args):
    candidate_cfg = config.get("candidate", {})
    session_cfg = config.get("session", {})
    default_id = f"CANDIDATE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    candidate_id = _sanitize_identifier(
        args.candidate_id or candidate_cfg.get("id") or session_cfg.get("candidate_id"),
        default_id
    )
    candidate_name = _safe_text(
        args.candidate_name or candidate_cfg.get("name") or session_cfg.get("candidate_name"),
        "Unknown Candidate"
    )
    exam_name = _safe_text(
        args.exam_name or candidate_cfg.get("exam") or session_cfg.get("exam"),
        "Proctored Session"
    )
    course_name = _safe_text(
        args.course_name or candidate_cfg.get("course") or session_cfg.get("course"),
        "General"
    )

    return {
        "id": candidate_id,
        "name": candidate_name,
        "exam": exam_name,
        "course": course_name,
    }


def _derive_current_activity(results, allow_mouth_violations):
    if results.get("objects_detected"):
        return "Object Detected"
    if not results.get("face_present", True):
        return "Face Missing"
    if results.get("multiple_faces"):
        return "Multiple Faces"
    if results.get("identity_mismatch"):
        return "Identity Mismatch"
    if allow_mouth_violations and results.get("mouth_moving"):
        return "Mouth Movement"
    if str(results.get("gaze_direction", "center")).lower() != "center":
        return "Looking Away"
    if results.get("voice_status") == "Detected":
        return "Audio Detected"
    return "Normal"


def _estimate_cheating_probability(results, allow_mouth_violations):
    score = 5
    if not results.get("face_present", True):
        score += 40
    if results.get("multiple_faces"):
        score += 45
    if results.get("objects_detected"):
        score += 50
    if results.get("identity_mismatch"):
        score += 50
    if allow_mouth_violations and results.get("mouth_moving"):
        score += 20
    if str(results.get("gaze_direction", "center")).lower() != "center":
        score += 15
    if results.get("voice_status") == "Detected":
        score += 20
    return max(0, min(int(score), 100))


def _write_session_status(status_file, payload):
    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _update_voice_state(audio_state, lock, status, status_file, details=None):
    payload = {
        "voice_status": status,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": details or ""
    }
    with lock:
        audio_state["voice_status"] = status
        audio_state["updated_at"] = payload["updated_at"]
        audio_state["details"] = payload["details"]

    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def start_audio_monitor_thread(config, alert_system, alert_logger, violation_logger, audio_state, state_lock):
    audio_cfg = config["detection"]["audio_monitoring"]
    if not audio_cfg.get("enabled", False):
        return None, None

    stop_event = threading.Event()
    status_file = os.path.join(_resolve_project_path(config["logging"]["log_path"]), "audio_status.json")

    def _audio_worker():
        sample_rate = 16000
        frame_ms = 30
        frame_samples = int(sample_rate * frame_ms / 1000)
        chunk_bytes = frame_samples * 2  # int16 mono => 2 bytes/sample

        vad = webrtcvad.Vad(3)
        pa = pyaudio.PyAudio()

        energy_threshold = _safe_int_threshold(
            audio_cfg.get("energy_threshold", 1200),
            1200
        )
        debounce_window = int(audio_cfg.get("debounce_window_frames", 20))
        speech_trigger_frames = int(audio_cfg.get("speech_trigger_frames", 6))
        loud_trigger_frames = int(audio_cfg.get("loud_trigger_frames", 8))
        violation_cooldown = float(audio_cfg.get("violation_cooldown_seconds", 4.0))

        speech_buffer = deque(maxlen=max(debounce_window, 1))
        loud_buffer = deque(maxlen=max(debounce_window, 1))
        last_violation_time = 0.0

        _update_voice_state(audio_state, state_lock, "Listening", status_file, "Audio monitor ready")

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=frame_samples
            )
        except Exception as e:
            _update_voice_state(audio_state, state_lock, "Error", status_file, f"Mic open failed: {e}")
            pa.terminate()
            return

        try:
            while not stop_event.is_set():
                try:
                    frame = stream.read(frame_samples, exception_on_overflow=False)
                except Exception:
                    continue

                if len(frame) < chunk_bytes:
                    continue

                # Strict speech signal using VAD mode 3.
                try:
                    is_speech = vad.is_speech(frame, sample_rate)
                except Exception:
                    is_speech = False

                # Loud non-speech detection for sudden noise spikes.
                rms_energy = audioop.rms(frame, 2)
                is_loud_noise = (rms_energy >= energy_threshold) and (not is_speech)

                speech_buffer.append(1 if is_speech else 0)
                loud_buffer.append(1 if is_loud_noise else 0)

                sustained_speech = sum(speech_buffer) >= speech_trigger_frames
                sustained_loud = sum(loud_buffer) >= loud_trigger_frames

                if sustained_speech or sustained_loud:
                    now_ts = datetime.now()
                    now_epoch = now_ts.timestamp()
                    if now_epoch - last_violation_time >= violation_cooldown:
                        last_violation_time = now_epoch
                        event_kind = "speech" if sustained_speech else "loud_noise"

                        violation_logger.log_violation(
                            "AUDIO_DETECTED",
                            now_ts.strftime("%Y%m%d_%H%M%S_%f"),
                            {
                                "event_kind": event_kind,
                                "rms_energy": rms_energy,
                                "energy_threshold": energy_threshold
                            }
                        )
                        if alert_logger:
                            alert_logger.log_alert("AUDIO_DETECTED", f"Sustained audio detected ({event_kind})")
                        if alert_system:
                            alert_system.speak_alert("VOICE_DETECTED")

                    _update_voice_state(audio_state, state_lock, "Detected", status_file, f"RMS={rms_energy}")
                else:
                    _update_voice_state(audio_state, state_lock, "Listening", status_file, f"RMS={rms_energy}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()
            _update_voice_state(audio_state, state_lock, "Stopped", status_file, "Audio monitor stopped")

    thread = threading.Thread(target=_audio_worker, daemon=True)
    thread.start()
    return thread, stop_event


def start_object_detection_thread(object_detector, shared_state, state_lock):
    stop_event = threading.Event()

    def _object_worker():
        last_processed_id = -1
        while not stop_event.is_set():
            with state_lock:
                frame_id = shared_state["frame_id"]
                frame = shared_state["latest_frame"]

            if frame is None or frame_id == last_processed_id:
                stop_event.wait(0.01)
                continue

            last_processed_id = frame_id

            # Requirement: run AI detection on downscaled frame (imgsz=640 path).
            h, w = frame.shape[:2]
            target_w = 640
            target_h = max(int(h * (target_w / max(w, 1))), 1)
            resized = cv2.resize(frame, (target_w, target_h))

            detected, labels = object_detector.detect_objects(resized, visualize=False)
            if detected is None:
                continue
            with state_lock:
                shared_state["objects_detected"] = bool(detected)
                shared_state["detected_objects"] = labels or []
                shared_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    thread = threading.Thread(target=_object_worker, daemon=True)
    thread.start()
    return thread, stop_event

def display_detection_results(frame, results):
    y_offset = 30
    line_height = 30
    
    # Status indicators
    status_items = [
        f"Face: {'Present' if results['face_present'] else 'Absent'}",
        f"Gaze: {results['gaze_direction']}",
        f"Eyes: {'Open' if results['eye_ratio'] > 0.25 else 'Closed'}",
        f"Mouth: {results.get('mouth_status', 'Moving' if results['mouth_moving'] else 'Still')}",
        f"Voice: {results.get('voice_status', 'Listening')}",
        f"Identity: {results.get('identity_status', 'Not started')}"
    ]
    
    # Alert indicators
    alert_items = []
    if results['multiple_faces']:
        alert_items.append("Multiple Faces Detected!")
    if results['objects_detected']:
        alert_items.append("Suspicious Object Detected!")

    # Display status
    for item in status_items:
        cv2.putText(frame, item, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height
    
    # Display alerts
    for item in alert_items:
        cv2.putText(frame, item, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += line_height
    
    # Timestamp
    cv2.putText(frame, results['timestamp'], 
               (frame.shape[1] - 250, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main(cli_args=None):
    config = load_config()
    args = cli_args or parse_args()

    config_session_mode = str(config.get("session", {}).get("mode", "exam")).strip().lower()
    session_mode = (args.mode or config_session_mode or "exam").lower()
    if session_mode not in {"exam", "interview"}:
        session_mode = "exam"

    source_override = _parse_video_source(args.source)
    if source_override is not None:
        config['video']['source'] = source_override
    if args.no_screen_recording:
        config['screen']['recording'] = False
    if args.disable_audio or session_mode == "interview":
        config['detection']['audio_monitoring']['enabled'] = False
    config['detection']['objects']['strict_mode'] = bool(args.strict_objects)

    # Policy toggles by session mode.
    allow_mouth_violations = session_mode == "exam"

    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violation_logger = ViolationLogger(config)
    report_generator = ReportGenerator(config)
    violation_logger.reset()

    student_info = _build_student_info(config, args)
    session_status_file = os.path.join(
        _resolve_project_path(config["logging"]["log_path"]),
        "session_status.json"
    )
    last_alert_time = None
    latest_face_detected = False
    latest_activity = "Initializing"
    latest_probability = 0
    peak_probability = 0

    
    # Initialize recorders
    video_recorder = VideoRecorder(config)
    screen_recorder = ScreenRecorder(config)
    
    # Initialize real-time audio monitoring state
    audio_state = {
        "voice_status": "Allowed" if session_mode == "interview" else "Disabled",
        "updated_at": None,
        "details": ""
    }
    audio_state_lock = threading.Lock()
    audio_thread = None
    audio_stop_event = None

    objects_enabled = not args.disable_objects
    object_thread = None
    object_stop_event = None
    object_state_lock = threading.Lock()
    object_state = {
        "frame_id": 0,
        "latest_frame": None,
        "objects_detected": False,
        "detected_objects": [],
        "last_update": None,
    }

    cap = None

    if config['detection']['audio_monitoring'].get('enabled', False):
        audio_thread, audio_stop_event = start_audio_monitor_thread(
            config,
            alert_system,
            alert_logger,
            violation_logger,
            audio_state,
            audio_state_lock
        )
    else:
        _update_voice_state(
            audio_state,
            audio_state_lock,
            "Allowed" if session_mode == "interview" else "Disabled",
            os.path.join(_resolve_project_path(config["logging"]["log_path"]), "audio_status.json"),
            f"session_mode={session_mode}"
        )

    _write_session_status(session_status_file, {
        "session_state": "starting",
        "mode": session_mode,
        "candidate": student_info,
        "face_detected": False,
        "current_activity": "Initializing",
        "cheating_probability": 0,
        "last_alert": None,
        "voice_status": audio_state.get("voice_status", "Unknown"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    try:
        if config['screen']['recording']:
            screen_recorder.start_recording()
        # Initialize detectors
        detectors = [
            FaceDetector(config),
            EyeTracker(config),
            MouthMonitor(config),
            MultiFaceDetector(config),
        ]
        identity_verifier = IdentityVerifier(config)
        object_detector = None
        if objects_enabled:
            object_detector = ObjectDetector(config)
            detectors.append(object_detector)
        
        for detector in detectors:
            if hasattr(detector, 'set_alert_logger'):
                detector.set_alert_logger(alert_logger)

        if object_detector is not None:
            object_thread, object_stop_event = start_object_detection_thread(
                object_detector,
                object_state,
                object_state_lock
            )

        # Start webcam recording
        video_recorder.start_recording()
        cap = cv2.VideoCapture(config['video']['source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['resolution'][1])
        object_prev_detected = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = {
                'face_present': False,
                'gaze_direction': 'Center',
                'eye_ratio': 0.3,
                'mouth_moving': False,
                'mouth_status': 'Allowed' if not allow_mouth_violations else 'Still',
                'multiple_faces': False,
                'objects_detected': False,
                'detected_objects': [],
                'voice_status': 'Listening',
                'identity_status': 'Enrolling',
                'identity_enrolled': False,
                'identity_distance': None,
                'identity_mismatch': False,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Perform detections
            results['face_present'] = detectors[0].detect_face(frame)
            results['gaze_direction'], results['eye_ratio'] = detectors[1].track_eyes(frame)
            if allow_mouth_violations:
                results['mouth_moving'] = detectors[2].monitor_mouth(frame)
                results['mouth_status'] = 'Moving' if results['mouth_moving'] else 'Still'
            else:
                results['mouth_moving'] = False
                results['mouth_status'] = 'Allowed'
            results['multiple_faces'] = detectors[3].detect_multiple_faces(frame)
            identity_result = identity_verifier.verify(frame)
            results['identity_status'] = identity_result['status']
            results['identity_enrolled'] = identity_result['enrolled']
            results['identity_distance'] = identity_result['distance']
            results['identity_mismatch'] = identity_result['identity_mismatch']

            if object_detector is not None:
                with object_state_lock:
                    object_state["frame_id"] += 1
                    object_state["latest_frame"] = frame.copy()

                with object_state_lock:
                    results['objects_detected'] = object_state["objects_detected"]
                    results['detected_objects'] = list(object_state["detected_objects"])

            with audio_state_lock:
                results['voice_status'] = audio_state.get('voice_status', 'Listening')

            object_triggered = results['objects_detected'] and not object_prev_detected
            object_prev_detected = results['objects_detected']
            triggered_violation = None

            if object_triggered:
                violation_type = "OBJECT_DETECTED"
                triggered_violation = violation_type
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {
                        'duration': '5+ seconds',
                        'detected_objects': results.get('detected_objects', []),
                        'frame': results
                    }
                )
                # alert_system.speak_alert("OBJECT_DETECTED")
            elif not results['face_present']:
                violation_type = "FACE_DISAPPEARED"
                triggered_violation = violation_type
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {'duration': '5+ seconds', 'frame': results}
                )
                # alert_system.speak_alert("FACE_DISAPPEARED")
            elif results['multiple_faces']:
                violation_type = "MULTIPLE_FACES"
                triggered_violation = violation_type
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {'duration': '5+ seconds', 'frame': results}
                )
                # alert_system.speak_alert("MULTIPLE_FACES")
            elif results['identity_mismatch']:
                violation_type = "IDENTITY_MISMATCH"
                triggered_violation = violation_type
                alert_system.speak_alert(violation_type)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {
                        'distance': results.get('identity_distance'),
                        'threshold': float(
                            config.get("detection", {})
                            .get("identity_verification", {})
                            .get("distance_threshold", 0.45)
                        ),
                        'frame': results
                    }
                )
            # elif results['gaze_direction'] != "Center":
            #     violation_type = "GAZE_AWAY"
            #     alert_system.speak_alert(violation_type)
                
            #     # Capture and log violation
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            #     violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
            #     violation_logger.log_violation(
            #         violation_type,
            #         timestamp,
            #         {'duration': '5+ seconds', 'frame': results}
            #     )
                # alert_system.speak_alert("GAZE_AWAY")
            elif results['mouth_moving'] and allow_mouth_violations:
                violation_type = "MOUTH_MOVING"
                triggered_violation = violation_type
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {'duration': '5+ seconds', 'frame': results}
                )
                # alert_system.speak_alert("MOUTH_MOVING")

            if triggered_violation:
                last_alert_time = results['timestamp']

            current_activity = _derive_current_activity(results, allow_mouth_violations)
            current_probability = _estimate_cheating_probability(results, allow_mouth_violations)
            latest_face_detected = bool(results.get("face_present", False))
            latest_activity = current_activity
            latest_probability = current_probability
            peak_probability = max(peak_probability, current_probability)

            _write_session_status(session_status_file, {
                "session_state": "running",
                "mode": session_mode,
                "candidate": student_info,
                "face_detected": latest_face_detected,
                "current_activity": latest_activity,
                "cheating_probability": latest_probability,
                "last_alert": last_alert_time,
                "voice_status": results.get("voice_status", "Unknown"),
                "latest_violation": triggered_violation,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            
            # Display and record
            display_detection_results(frame, results)
            video_recorder.record_frame(frame)
            
            # Show preview only in interactive mode
            if not args.headless:
                cv2.imshow('Exam Proctoring', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    finally:
        if object_stop_event is not None:
            object_stop_event.set()
        if object_thread is not None and object_thread.is_alive():
            object_thread.join(timeout=2)

        if audio_stop_event is not None:
            audio_stop_event.set()
        if audio_thread is not None and audio_thread.is_alive():
            audio_thread.join(timeout=2)

        violations = violation_logger.get_violations()
        report_path = report_generator.generate_report(student_info, violations)
        if report_path:
            print(f"Report generated: {report_path}")
        else:
            print("Report generation failed. Check logs for details.")

        _write_session_status(session_status_file, {
            "session_state": "ended",
            "mode": session_mode,
            "candidate": student_info,
            "face_detected": latest_face_detected,
            "current_activity": latest_activity if latest_activity != "Initializing" else "Session Ended",
            "cheating_probability": max(latest_probability, peak_probability),
            "last_alert": last_alert_time,
            "voice_status": audio_state.get("voice_status", "Stopped"),
            "total_violations": len(violations),
            "report_path": report_path,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        if config['screen']['recording']:
            screen_data = screen_recorder.stop_recording()
            if screen_data:
                print(f"Screen recording saved: {screen_data['filename']}")
            else:
                print("Screen recording was not started, nothing to save.")

        video_data = video_recorder.stop_recording()
        if video_data:
            print(f"Webcam recording saved: {video_data['filename']}")
        else:
            print("Webcam recording was not started or not available.")
        
        if cap is not None and cap.isOpened():
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
