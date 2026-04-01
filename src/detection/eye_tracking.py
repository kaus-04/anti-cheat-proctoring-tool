import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import importlib
from collections import deque


def _get_face_mesh_module():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh

    # Some installs don't expose `solutions` as an attribute on the root package
    # but still provide the module path.
    for module_name in ("mediapipe.solutions.face_mesh", "mediapipe.python.solutions.face_mesh"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue

    raise ImportError(
        "Could not import FaceMesh from mediapipe. Tried mp.solutions.face_mesh, "
        "mediapipe.solutions.face_mesh, and mediapipe.python.solutions.face_mesh."
    )


class EyeTracker:
    def __init__(self, config):
        self.config = config
        self.eye_threshold = config['detection']['eyes']['gaze_threshold']
        self.alert_logger = None

        self.gaze_direction = "center"
        self.eye_ratio = 0.3

        self.enabled = False
        try:
            self.mp_face_mesh = _get_face_mesh_module()
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            self.enabled = True
        except Exception as e:
            print(f"Warning: EyeTracker disabled due to MediaPipe init error: {e}")
            self.mp_face_mesh = None
            self.face_mesh = None
        self.last_gaze_change = datetime.now()
        self.gaze_direction = "center"  # Default value
        self.eye_ratio = 0.3  # Default open eye ratio
        self.gaze_changes = 0
        self.alert_logger = None
        
        # Landmark indices for left and right eyes
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.LEFT_EYE_CORNERS = (33, 133)
        self.RIGHT_EYE_CORNERS = (362, 263)
        # Iris landmarks (requires refine_landmarks=True).
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        self.iris_history = deque(maxlen=5)
        self.head_history = deque(maxlen=5)
        self.fused_history = deque(maxlen=5)
        
        # For EAR (Eye Aspect Ratio) calculation
        self.EYE_ASPECT_RATIO_THRESH = 0.3
        self.EYE_ASPECT_RATIO_CONSEC_FRAMES = 3

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def _calculate_ear(self, eye_points):
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def _landmark_xy(self, face_landmarks, idx, frame_w, frame_h):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h], dtype=np.float32)

    def _iris_horizontal_ratio(self, face_landmarks, iris_indices, eye_corners, frame_w, frame_h):
        iris_points = np.array(
            [self._landmark_xy(face_landmarks, i, frame_w, frame_h) for i in iris_indices],
            dtype=np.float32
        )
        iris_center_x = float(np.mean(iris_points[:, 0]))

        c1 = self._landmark_xy(face_landmarks, eye_corners[0], frame_w, frame_h)
        c2 = self._landmark_xy(face_landmarks, eye_corners[1], frame_w, frame_h)
        min_x = float(min(c1[0], c2[0]))
        max_x = float(max(c1[0], c2[0]))
        width = max_x - min_x
        if width < 1e-6:
            return 0.5

        ratio = (iris_center_x - min_x) / width
        return float(np.clip(ratio, 0.0, 1.0))

    def track_eyes(self, frame):
        if not self.enabled or self.face_mesh is None:
            return self.gaze_direction, self.eye_ratio

        try:
            # Convert frame to RGB and process
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return self.gaze_direction, self.eye_ratio  # Return last known values
            
            face_landmarks = results.multi_face_landmarks[0]
            frame_h, frame_w = frame.shape[:2]
            
            # Get eye landmarks in pixel coordinates
            left_eye_coords = np.array([(face_landmarks.landmark[i].x * frame_w, 
                                       face_landmarks.landmark[i].y * frame_h) 
                                      for i in self.LEFT_EYE_INDICES])
            
            right_eye_coords = np.array([(face_landmarks.landmark[i].x * frame_w, 
                                        face_landmarks.landmark[i].y * frame_h) 
                                       for i in self.RIGHT_EYE_INDICES])
            
            # Calculate Eye Aspect Ratio (EAR) for both eyes
            left_ear = self._calculate_ear(left_eye_coords)
            right_ear = self._calculate_ear(right_eye_coords)
            self.eye_ratio = (left_ear + right_ear) / 2.0
            
            # Head-pose cue (normalized): positive -> right, negative -> left.
            left_eye_center = np.mean(left_eye_coords, axis=0)
            right_eye_center = np.mean(right_eye_coords, axis=0)
            eye_mid_x = float((left_eye_center[0] + right_eye_center[0]) / 2.0)
            inter_eye_dist = float(np.linalg.norm(left_eye_center - right_eye_center))
            nose_tip_x = float(face_landmarks.landmark[4].x * frame_w)
            head_score = 0.0
            if inter_eye_dist > 1e-6:
                head_score = (eye_mid_x - nose_tip_x) / inter_eye_dist
            self.head_history.append(head_score)
            smooth_head = float(np.mean(self.head_history))

            # Iris cue (normalized): centered near 0, side gaze moves away from 0.
            iris_score = 0.0
            try:
                left_ratio = self._iris_horizontal_ratio(
                    face_landmarks,
                    self.LEFT_IRIS_INDICES,
                    self.LEFT_EYE_CORNERS,
                    frame_w,
                    frame_h
                )
                right_ratio = self._iris_horizontal_ratio(
                    face_landmarks,
                    self.RIGHT_IRIS_INDICES,
                    self.RIGHT_EYE_CORNERS,
                    frame_w,
                    frame_h
                )
                avg_ratio = (left_ratio + right_ratio) / 2.0
                iris_score = avg_ratio - 0.5
            except Exception:
                # Keep iris contribution neutral if iris landmarks are unavailable.
                iris_score = 0.0

            self.iris_history.append(iris_score)
            smooth_iris = float(np.mean(self.iris_history))

            # Fuse both: head has slightly higher weight for stability.
            fused_score = (0.6 * smooth_head) + (0.4 * smooth_iris)
            self.fused_history.append(fused_score)
            smooth_fused = float(np.mean(self.fused_history))

            new_gaze = "center"
            if smooth_fused < -0.08:
                new_gaze = "left"
            elif smooth_fused > 0.08:
                new_gaze = "right"
            
            # Update gaze changes
            current_time = datetime.now()
            if new_gaze != self.gaze_direction:
                self.gaze_changes += 1
                self.gaze_direction = new_gaze
                self.last_gaze_change = current_time
                
            # Check for excessive eye movement
            if (self.gaze_changes > 3 and 
                (current_time - self.last_gaze_change).total_seconds() < 2 and
                self.alert_logger):
                self.alert_logger.log_alert(
                    "EYE_MOVEMENT",
                    "Excessive eye movement detected"
                )
                self.gaze_changes = 0
            
            return self.gaze_direction, self.eye_ratio
            
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "EYE_TRACKING_ERROR",
                    f"Error in eye tracking: {str(e)}"
                )
            return self.gaze_direction, self.eye_ratio  # Return last known values
