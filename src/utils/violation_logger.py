import os
import json
from datetime import datetime

class ViolationLogger:
    def __init__(self, config):
        self.log_file = os.path.join(config['global']['output_path'], "violations.json")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.violations = []

    def reset(self):
        """Start a fresh session log."""
        self.violations = []
        self._save_to_file()
        
    def log_violation(self, violation_type, timestamp=None, metadata=None):
        """Logs a violation with timestamp and metadata"""
        entry = {
            'type': violation_type,
            'timestamp': timestamp or datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.violations.append(entry)
        self._save_to_file()
        
    def _save_to_file(self):
        """Saves violations to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.violations, f, indent=2)
            
    def get_violations(self):
        """Returns all logged violations"""
        return self.violations
