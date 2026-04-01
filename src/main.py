import cv2
import yaml
import argparse
import os
import json
import audioop
import pyaudio
import webrtcvad
import threading
from collections import deque
from pathlib import Path
from datetime import datetime
from detection.face_detection import FaceDetector
from detection.eye_tracking import EyeTracker
from detection.mouth_detection import MouthMonitor
from detection.object_detection import ObjectDetector
from detection.multi_face import MultiFaceDetector
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
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--disable-audio", action="store_true", help="Disable audio monitoring")
    parser.add_argument(
        "--no-screen-recording",
        action="store_true",
        help="Disable screen recording for this run"
    )
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

def display_detection_results(frame, results):
    y_offset = 30
    line_height = 30
    
    # Status indicators
    status_items = [
        f"Face: {'Present' if results['face_present'] else 'Absent'}",
        f"Gaze: {results['gaze_direction']}",
        f"Eyes: {'Open' if results['eye_ratio'] > 0.25 else 'Closed'}",
        f"Mouth: {'Moving' if results['mouth_moving'] else 'Still'}",
        f"Voice: {results.get('voice_status', 'Listening')}"
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

    source_override = _parse_video_source(args.source)
    if source_override is not None:
        config['video']['source'] = source_override
    if args.no_screen_recording:
        config['screen']['recording'] = False
    if args.disable_audio:
        config['detection']['audio_monitoring']['enabled'] = False

    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violation_logger = ViolationLogger(config)
    report_generator = ReportGenerator(config)

    student_info = {
        'id': 'STUDENT_001',
        'name': 'John Doe',
        'exam': 'Final Examination',
        'course': 'Computer Science 101'
    }

    
    # Initialize recorders
    video_recorder = VideoRecorder(config)
    screen_recorder = ScreenRecorder(config)
    
    # Initialize real-time audio monitoring state
    audio_state = {
        "voice_status": "Disabled",
        "updated_at": None,
        "details": ""
    }
    audio_state_lock = threading.Lock()
    audio_thread = None
    audio_stop_event = None

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

    try:
        if config['screen']['recording']:
            screen_recorder.start_recording()
        # Initialize detectors
        detectors = [
            FaceDetector(config),
            EyeTracker(config),
            MouthMonitor(config),
            MultiFaceDetector(config),
            ObjectDetector(config),
        ]
        
        for detector in detectors:
            if hasattr(detector, 'set_alert_logger'):
                detector.set_alert_logger(alert_logger)

        # Start webcam recording
        video_recorder.start_recording()
        cap = cv2.VideoCapture(config['video']['source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['resolution'][1])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = {
                'face_present': False,
                'gaze_direction': 'Center',
                'eye_ratio': 0.3,
                'mouth_moving': False,
                'multiple_faces': False,
                'objects_detected': False,
                'voice_status': 'Listening',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Perform detections
            results['face_present'] = detectors[0].detect_face(frame)
            results['gaze_direction'], results['eye_ratio'] = detectors[1].track_eyes(frame)
            results['mouth_moving'] = detectors[2].monitor_mouth(frame)
            results['multiple_faces'] = detectors[3].detect_multiple_faces(frame)
            results['objects_detected'] = detectors[4].detect_objects(frame)
            with audio_state_lock:
                results['voice_status'] = audio_state.get('voice_status', 'Listening')

            if not results['face_present']:
                violation_type = "FACE_DISAPPEARED"
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
            elif results['objects_detected']:
                violation_type = "OBJECT_DETECTED"
                alert_system.speak_alert(violation_type)
                
                # Capture and log violation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {'duration': '5+ seconds', 'frame': results}
                )
                # alert_system.speak_alert("OBJECT_DETECTED")
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
            elif results['mouth_moving']:
                violation_type = "MOUTH_MOVING"
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

            
            # Display and record
            display_detection_results(frame, results)
            video_recorder.record_frame(frame)
            
            # Show preview only in interactive mode
            if not args.headless:
                cv2.imshow('Exam Proctoring', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    finally:
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
