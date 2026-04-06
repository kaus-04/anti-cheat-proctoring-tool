import os
import tempfile
from gtts import gTTS
import pygame
import threading
import time

class AlertSystem:
    def __init__(self, config):
        pygame.mixer.init()
        self.config = config
        self.alert_cooldown = config['logging']['alert_cooldown']
        self.last_alert_time = {}
        self._audio_lock = threading.Lock()
        
        # Alert messages database
        self.alerts = {
            "FACE_DISAPPEARED": "Please look at the screen",
            "FACE_REAPPEARED": "Thank you for looking at the screen",
            "MULTIPLE_FACES": "We detected multiple people",
            "OBJECT_DETECTED": "Unauthorized object detected",
            "IDENTITY_MISMATCH": "Identity mismatch detected",
            "GAZE_AWAY": "Please focus on your screen",
            "MOUTH_MOVING": "Please maintain silence during exam",
            "SPEECH_VIOLATION": "Speaking during exam is not allowed",
            "VOICE_DETECTED": "We detected voice, Please maintain silence during the exam",
        }
        
    def _can_alert(self, alert_type):
        """Check if enough time has passed since last alert"""
        current_time = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)
        return (current_time - last_time) >= self.alert_cooldown
        
    def speak_alert(self, alert_type):
        """Convert text to speech and play it"""
        if not self._can_alert(alert_type):
            return
            
        self.last_alert_time[alert_type] = time.time()
        
        def _play_audio():
            temp_path = None
            try:
                if alert_type in self.alerts:
                    # Serialize mixer access to avoid races across alert threads.
                    with self._audio_lock:
                        # Generate speech
                        tts = gTTS(text=self.alerts[alert_type], lang='en')

                        # Save temporary audio file
                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                            temp_path = fp.name
                        tts.save(temp_path)

                        # Play audio
                        pygame.mixer.music.load(temp_path)
                        pygame.mixer.music.play()

                        # Wait until playback finishes
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
            except Exception as e:
                print(f"Audio alert failed: {str(e)}")
            finally:
                # Ensure player releases handle before deleting file (important on Windows).
                try:
                    pygame.mixer.music.stop()
                    try:
                        pygame.mixer.music.unload()
                    except Exception:
                        pass
                except Exception:
                    pass

                if temp_path and os.path.exists(temp_path):
                    for _ in range(10):
                        try:
                            os.unlink(temp_path)
                            break
                        except PermissionError:
                            time.sleep(0.1)
        
        # Run in separate thread to avoid blocking
        threading.Thread(target=_play_audio, daemon=True).start()
