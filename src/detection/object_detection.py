# import cv2
# import torch
# from ultralytics import YOLO
# from datetime import datetime

# class ObjectDetector:
#     def __init__(self, config):
#         self.config = config['detection']['objects']
#         self.model = None
#         self.class_map = {
#             73: 'book',
#             67: 'cell phone'
#         }
#         self.alert_logger = None
#         self.detection_interval = self.config['detection_interval']
#         self.frame_count = 0
#         self._initialize_model()

#     def _initialize_model(self):
#         """Safely initialize the YOLO model"""
#         try:
#             self.model = YOLO('models/yolov8n.pt')
#             # Warm up the model
#             dummy_input = torch.zeros((1, 3, 640, 640))
#             self.model(dummy_input)
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize object detector: {str(e)}")

#     def set_alert_logger(self, alert_logger):
#         self.alert_logger = alert_logger

#     def detect_objects(self, frame, visualize=False):
#         """Detect forbidden objects in frame"""
#         self.frame_count += 1
#         if self.frame_count % self.detection_interval != 0:
#             return False
            
#         try:
#             results = self.model(frame)
#             detected = False
            
#             for result in results:
#                 for box in result.boxes:
#                     cls = int(box.cls)
#                     conf = float(box.conf)
                    
#                     if cls in self.class_map and conf > self.config['min_confidence']:
#                         detected = True
#                         label = self.class_map[cls]
                        
#                         if self.alert_logger:
#                             self.alert_logger.log_alert(
#                                 "FORBIDDEN_OBJECT",
#                                 f"Detected {label} with confidence {conf:.2f}"
#                             )
                        
#                         if visualize:
#                             x1, y1, x2, y2 = map(int, box.xyxy[0])
#                             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                             cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
#             return detected
            
#         except Exception as e:
#             if self.alert_logger:
#                 self.alert_logger.log_alert(
#                     "OBJECT_DETECTION_ERROR",
#                     f"Object detection failed: {str(e)}"
#                 )
#             return False


import cv2
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

class ObjectDetector:
    def __init__(self, config):
        self.config = config['detection']['objects']
        self.model = None
        self.class_aliases = {
            "cellphone": "cell phone",
            "mobile phone": "cell phone",
            "smartphone": "cell phone"
        }
        raw_targets = self.config.get('target_objects', ['book', 'cell phone'])
        self.target_names = set(self._normalize_name(name) for name in raw_targets)
        raw_class_conf = self.config.get('class_min_confidence', {})
        self.class_min_confidence = {
            self._normalize_name(name): float(value)
            for name, value in raw_class_conf.items()
        }
        self.class_map = {}
        self.alert_logger = None
        self.detection_interval = int(self.config.get('detection_interval', 1))
        self.frame_count = 0
        self._initialize_model()
        self.last_detection_time = datetime.now()

    def _normalize_name(self, name):
        normalized = str(name).lower().strip()
        return self.class_aliases.get(normalized, normalized)

    def _initialize_model(self):
        """Initialize optimized YOLO model"""
        try:
            # Prefer local model if present; otherwise use ultralytics default resolution path.
            local_model = Path('models/yolov8n.pt')
            model_path = str(local_model) if local_model.exists() else 'yolov8n.pt'
            self.model = YOLO(model_path)

            names = self.model.names if isinstance(self.model.names, dict) else {}
            for class_id, class_name in names.items():
                normalized = self._normalize_name(class_name)
                if normalized in self.target_names:
                    self.class_map[int(class_id)] = normalized

            if not self.class_map and self.alert_logger:
                self.alert_logger.log_alert(
                    "OBJECT_DETECTION_WARNING",
                    f"Target classes not found in model: {sorted(self.target_names)}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize object detector: {str(e)}")

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def detect_objects(self, frame, visualize=False):
        """Object detection with frame skipping and class filtering"""
        self.frame_count += 1
        if self.frame_count % max(self.detection_interval, 1) != 0:
            return False

        current_time = datetime.now()
        time_since_last = (current_time - self.last_detection_time).total_seconds()
        
        # Skip detection if not enough time has passed
        if time_since_last < (1.0 / max(float(self.config.get('max_fps', 5)), 1.0)):
            return False
            
        try:
            orig_h, orig_w = frame.shape[:2]
            imgsz = int(self.config.get('imgsz', 640))
            conf_threshold = float(self.config.get('min_confidence', 0.35))

            # Run inference directly on original frame for better small-object recall.
            results = self.model(frame, verbose=False, imgsz=imgsz, conf=conf_threshold)
            
            detected = False
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    
                    if cls not in self.class_map:
                        continue

                    label = self.class_map[cls]
                    class_threshold = self.class_min_confidence.get(label, conf_threshold)

                    if conf >= class_threshold:
                        detected = True
                        
                        if self.alert_logger:
                            self.alert_logger.log_alert(
                                "FORBIDDEN_OBJECT",
                                f"Detected {label} with confidence {conf:.2f} (thr {class_threshold:.2f})"
                            )
                        
                        if visualize:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            self.last_detection_time = current_time
            return detected
            
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "OBJECT_DETECTION_ERROR",
                    f"Object detection failed: {str(e)}"
                )
            return False
