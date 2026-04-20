import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN # type: ignore


class IdentityVerifier:
    """Verifies that the same enrolled candidate remains in frame."""

    def __init__(self, config):
        identity_cfg = (
            config.get("detection", {})
            .get("identity_verification", {})
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.available = True
        self.init_error = None
        try:
            self.face_detector = MTCNN(
                keep_all=True,
                post_process=False,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                device=self.device,
            )
            self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        except Exception as exc:
            self.available = False
            self.init_error = str(exc)
            self.face_detector = None
            self.embedder = None

        self.enrollment_samples_target = int(identity_cfg.get("enrollment_samples", 24))
        self.check_interval = int(identity_cfg.get("check_interval", 10))
        self.distance_threshold = float(identity_cfg.get("distance_threshold", 0.45))
        self.mismatch_consecutive_limit = int(identity_cfg.get("mismatch_consecutive", 3))
        self.min_face_confidence = float(identity_cfg.get("min_face_confidence", 0.90))

        self._frame_count = 0
        self._enrollment_samples = []
        self._baseline_embedding = None
        self._consecutive_mismatches = 0
        self._mismatch_active = False
        self._last_distance = None

    @staticmethod
    def _normalize(embedding_vector):
        norm = np.linalg.norm(embedding_vector)
        if norm <= 1e-10:
            return embedding_vector
        return embedding_vector / norm

    def _extract_single_face_embedding(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            faces, probs = self.face_detector(rgb_frame, return_prob=True)

        if faces is None or probs is None:
            return None

        valid_indices = [i for i, prob in enumerate(probs) if prob is not None and prob >= self.min_face_confidence]
        if len(valid_indices) != 1:
            return None

        face_tensor = faces[valid_indices[0]]
        if face_tensor.ndim != 3:
            return None

        with torch.no_grad():
            embedding = self.embedder(face_tensor.unsqueeze(0).to(self.device))

        embedding_np = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return self._normalize(embedding_np)

    def _cosine_distance(self, lhs, rhs):
        return float(1.0 - np.dot(lhs, rhs))

    @property
    def enrolled(self):
        return self._baseline_embedding is not None

    def verify(self, frame):
        """
        Returns:
            dict with keys:
              identity_mismatch (bool): True only on mismatch activation edge.
              enrolled (bool): enrollment completion flag.
              status (str): current identity status for UI/debug.
              distance (float|None): latest cosine distance when available.
              consecutive_mismatches (int): mismatch streak length.
        """
        self._frame_count += 1

        if not self.available:
            return {
                "identity_mismatch": False,
                "enrolled": False,
                "status": "Unavailable",
                "distance": None,
                "consecutive_mismatches": 0,
                "error": self.init_error,
            }

        if self._frame_count % max(self.check_interval, 1) != 0:
            return {
                "identity_mismatch": False,
                "enrolled": self.enrolled,
                "status": (
                    f"Enrolling {len(self._enrollment_samples)}/{self.enrollment_samples_target}"
                    if not self.enrolled
                    else "Verified"
                ),
                "distance": self._last_distance,
                "consecutive_mismatches": self._consecutive_mismatches,
            }

        embedding = self._extract_single_face_embedding(frame)

        if not self.enrolled:
            if embedding is not None:
                self._enrollment_samples.append(embedding)

                if len(self._enrollment_samples) >= self.enrollment_samples_target:
                    baseline = np.mean(np.stack(self._enrollment_samples, axis=0), axis=0)
                    self._baseline_embedding = self._normalize(baseline.astype(np.float32))
                    self._enrollment_samples = []
                    return {
                        "identity_mismatch": False,
                        "enrolled": True,
                        "status": "Enrolled",
                        "distance": None,
                        "consecutive_mismatches": 0,
                    }

            return {
                "identity_mismatch": False,
                "enrolled": False,
                "status": f"Enrolling {len(self._enrollment_samples)}/{self.enrollment_samples_target}",
                "distance": None,
                "consecutive_mismatches": 0,
            }

        if embedding is None:
            self._consecutive_mismatches = 0
            self._mismatch_active = False
            self._last_distance = None
            return {
                "identity_mismatch": False,
                "enrolled": True,
                "status": "Check Paused (No clear single face)",
                "distance": None,
                "consecutive_mismatches": 0,
            }

        distance = self._cosine_distance(self._baseline_embedding, embedding)
        self._last_distance = distance

        if distance > self.distance_threshold:
            self._consecutive_mismatches += 1
        else:
            self._consecutive_mismatches = 0
            self._mismatch_active = False

        mismatch_now = self._consecutive_mismatches >= self.mismatch_consecutive_limit
        mismatch_trigger = mismatch_now and (not self._mismatch_active)
        if mismatch_now:
            self._mismatch_active = True

        status = (
            f"Mismatch {self._consecutive_mismatches}/{self.mismatch_consecutive_limit}"
            if distance > self.distance_threshold
            else "Verified"
        )

        return {
            "identity_mismatch": mismatch_trigger,
            "enrolled": True,
            "status": status,
            "distance": distance,
            "consecutive_mismatches": self._consecutive_mismatches,
        }
