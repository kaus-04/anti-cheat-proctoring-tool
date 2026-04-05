from flask import Flask, render_template, jsonify, request, send_from_directory
import yaml
from datetime import datetime
from pathlib import Path
import subprocess
import threading
import uuid
import sys
import json
import tempfile
import inspect
import re
from collections import Counter, defaultdict
from werkzeug.utils import secure_filename
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

try:
    from copydetect import CopyDetector  # type: ignore
    COPYDETECT_AVAILABLE = True
except Exception:
    CopyDetector = None
    COPYDETECT_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB upload limit

# Load configuration
with open(PROJECT_ROOT / 'config' / 'config.yaml') as f:
    config = yaml.safe_load(f)

JOBS = {}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _resolve_config_path(path_text):
    path_obj = Path(path_text)
    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj)


def _is_allowed_video(filename):
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


REPORTS_DIR = _resolve_config_path(config['reporting']['output_dir'])
IMAGES_DIR = REPORTS_DIR / "images"
VIOLATIONS_FILE = _resolve_config_path(config['global']['output_path']) / "violations.json"
LOG_DIR = _resolve_config_path(config['logging']['log_path'])
VOICE_STATUS_FILE = LOG_DIR / "audio_status.json"
SEVERITY_LEVELS = config.get('reporting', {}).get('severity_levels', {})

REFERENCE_SOLUTIONS_DIR = PROJECT_ROOT / "src" / "reference_solutions"
REFERENCE_SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

AI_DETECTOR_MODEL = "roberta-base-openai-detector"
AI_DETECTOR_PIPELINE = None
AI_DETECTOR_LOAD_ERROR = None
if TRANSFORMERS_AVAILABLE:
    try:
        AI_DETECTOR_PIPELINE = pipeline(
            "text-classification",
            model=AI_DETECTOR_MODEL,
            tokenizer=AI_DETECTOR_MODEL,
        )
    except Exception as exc:
        AI_DETECTOR_LOAD_ERROR = str(exc)
else:
    AI_DETECTOR_LOAD_ERROR = "transformers library not installed"


def _safe_relative_path(base_dir, target_path):
    base = base_dir.resolve()
    target = Path(target_path).resolve()
    target.relative_to(base)
    return str(target.relative_to(base)).replace("\\", "/")


def _latest_file(path, patterns):
    files = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _parse_violation_time(value):
    if not value:
        return None
    for fmt in ("%Y%m%d_%H%M%S_%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _normalize_code_for_similarity(text):
    text = text.lower()
    text = re.sub(r"#[^\n]*", " ", text)
    text = re.sub(r"\"\"\"[\s\S]*?\"\"\"", " ", text)
    text = re.sub(r"\'\'\'[\s\S]*?\'\'\'", " ", text)
    text = re.sub(r"\b[_a-zA-Z][_a-zA-Z0-9]*\b", "id", text)
    text = re.sub(r"\d+", "num", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fallback_similarity_percent(submission_code):
    submission_tokens = set(_normalize_code_for_similarity(submission_code).split())
    if not submission_tokens:
        return 0.0

    best = 0.0
    for ref_file in REFERENCE_SOLUTIONS_DIR.rglob("*"):
        if not ref_file.is_file():
            continue
        try:
            ref_text = ref_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        ref_tokens = set(_normalize_code_for_similarity(ref_text).split())
        if not ref_tokens:
            continue
        score = (len(submission_tokens & ref_tokens) / max(len(submission_tokens | ref_tokens), 1)) * 100.0
        best = max(best, score)
    return round(best, 2)


def _extract_percentages_from_html(path):
    try:
        html = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    values = []
    for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", html):
        try:
            values.append(float(m))
        except ValueError:
            continue
    return [v for v in values if 0.0 <= v <= 100.0]


def _run_copydetect(submission_file, tmp_dir):
    if not COPYDETECT_AVAILABLE:
        return None, "copydetect library not installed"

    ref_files_exist = any(p.is_file() for p in REFERENCE_SOLUTIONS_DIR.rglob("*"))
    if not ref_files_exist:
        return 0.0, "reference_solutions directory is empty"

    out_file = Path(tmp_dir) / "copydetect_report.html"

    kwargs = {}
    sig = inspect.signature(CopyDetector)
    params = sig.parameters

    if "test_dirs" in params:
        kwargs["test_dirs"] = [str(Path(submission_file).parent)]
    if "ref_dirs" in params:
        kwargs["ref_dirs"] = [str(REFERENCE_SOLUTIONS_DIR)]
    if "out_file" in params:
        kwargs["out_file"] = str(out_file)
    if "autoopen" in params:
        kwargs["autoopen"] = False
    if "silent" in params:
        kwargs["silent"] = True

    detector = CopyDetector(**kwargs)

    ran = False
    for method_name in ("run", "analyze", "compare", "generate_report"):
        if hasattr(detector, method_name):
            getattr(detector, method_name)()
            ran = True
            break

    percentages = _extract_percentages_from_html(out_file)
    if percentages:
        return round(max(percentages), 2), None

    code_text = Path(submission_file).read_text(encoding="utf-8", errors="ignore")
    if ran:
        return _fallback_similarity_percent(code_text), "copydetect output not parseable; used fallback scoring"
    return _fallback_similarity_percent(code_text), "copydetect runner method not found; used fallback scoring"


def _ai_probability_percent(code_text):
    if AI_DETECTOR_PIPELINE is None:
        return 0.0, f"AI detector unavailable: {AI_DETECTOR_LOAD_ERROR or 'unknown error'}"

    sample = code_text[:4000]
    try:
        output = AI_DETECTOR_PIPELINE(sample, truncation=True)
    except Exception as exc:
        return 0.0, f"AI detector failed: {exc}"

    result = output[0]
    candidates = result if isinstance(result, list) else [result]

    ai_prob = 0.0
    for c in candidates:
        label = str(c.get("label", "")).lower()
        score = float(c.get("score", 0.0))

        if any(k in label for k in ("ai", "generated", "fake", "label_1")):
            ai_prob = max(ai_prob, score)
        elif any(k in label for k in ("human", "real", "label_0")):
            ai_prob = max(ai_prob, 1.0 - score)

    return round(ai_prob * 100.0, 2), None


def _risk_level(plagiarism_score, ai_probability):
    combined = max(plagiarism_score, ai_probability)
    if combined >= 70:
        return "High"
    if combined >= 40:
        return "Medium"
    return "Low"


def _build_summary_payload(violations):
    by_type = Counter(v.get("type", "UNKNOWN") for v in violations)
    severity_points = {
        vtype: count * int(SEVERITY_LEVELS.get(vtype, 1))
        for vtype, count in by_type.items()
    }

    minute_counts = defaultdict(int)
    for violation in violations:
        dt = _parse_violation_time(violation.get("timestamp"))
        if dt:
            minute_counts[dt.strftime("%H:%M")] += 1

    ordered_minutes = sorted(minute_counts.items(), key=lambda x: x[0])

    return {
        "total": len(violations),
        "by_type": dict(by_type),
        "severity_points_by_type": severity_points,
        "violations_by_minute": {
            "labels": [k for k, _ in ordered_minutes],
            "values": [v for _, v in ordered_minutes]
        }
    }


def _build_artifacts_payload():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    latest_report = _latest_file(REPORTS_DIR, ["*.pdf", "*.html"])
    latest_timeline = _latest_file(IMAGES_DIR, ["timeline_*.png"])
    latest_heatmap = _latest_file(IMAGES_DIR, ["heatmap_*.png"])

    payload = {
        "latest_report": None,
        "timeline_image_url": None,
        "heatmap_image_url": None
    }

    if latest_report:
        rel = _safe_relative_path(REPORTS_DIR, latest_report)
        payload["latest_report"] = {
            "name": latest_report.name,
            "download_url": f"/download/report/{rel}"
        }
    if latest_timeline:
        rel = _safe_relative_path(IMAGES_DIR, latest_timeline)
        payload["timeline_image_url"] = f"/artifacts/images/{rel}"
    if latest_heatmap:
        rel = _safe_relative_path(IMAGES_DIR, latest_heatmap)
        payload["heatmap_image_url"] = f"/artifacts/images/{rel}"

    return payload


def _run_analysis_job(job_id, video_path):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "main.py"),
        "--source",
        str(video_path),
        "--headless",
        "--disable-audio",
        "--no-screen-recording",
    ]

    output_lines = []
    report_path = None
    report_download_url = None

    JOBS[job_id].update({
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds")
    })

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            output_lines.append(line)
            if line.startswith("Report generated: "):
                report_path = line.replace("Report generated: ", "", 1).strip()

        return_code = process.wait()
        if report_path and report_path != "None":
            try:
                rel = _safe_relative_path(REPORTS_DIR, report_path)
                report_download_url = f"/download/report/{rel}"
            except Exception:
                report_download_url = None

        JOBS[job_id].update({
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "report_path": report_path,
            "report_download_url": report_download_url,
            "log_tail": output_lines[-100:],
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": _build_artifacts_payload()
        })
    except Exception as exc:
        JOBS[job_id].update({
            "status": "failed",
            "return_code": -1,
            "error": str(exc),
            "log_tail": output_lines[-100:],
            "finished_at": datetime.now().isoformat(timespec="seconds")
        })


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/alerts')
def get_alerts():
    log_dir = _resolve_config_path(config['logging']['log_path'])
    log_file = log_dir / "alerts.log"
    alerts = []

    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            alerts = [line.strip() for line in f.readlines()[-10:]]

    return jsonify(alerts)


@app.route('/api/stats')
def get_stats():
    voice_status = "Unknown"
    if VOICE_STATUS_FILE.exists():
        try:
            with open(VOICE_STATUS_FILE, "r", encoding="utf-8") as f:
                voice_status = (json.load(f) or {}).get("voice_status", "Unknown")
        except Exception:
            voice_status = "Unknown"

    return jsonify({
        'face_detected': True,
        'current_activity': 'Normal',
        'cheating_probability': 15,
        'last_alert': datetime.now().strftime("%H:%M:%S"),
        'voice_status': voice_status
    })


@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"ok": False, "error": "No video file provided."}), 400

    file = request.files['video']
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "No selected file."}), 400
    if not _is_allowed_video(file.filename):
        return jsonify({"ok": False, "error": "Unsupported file type."}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = secure_filename(file.filename)
    saved_path = UPLOADS_DIR / f"{timestamp}_{safe_name}"
    file.save(saved_path)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "video_path": str(saved_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_path": None,
        "report_download_url": None,
        "log_tail": []
    }

    threading.Thread(
        target=_run_analysis_job,
        args=(job_id, saved_path),
        daemon=True
    ).start()

    return jsonify({"ok": True, "job_id": job_id})


@app.route('/api/job/<job_id>')
def get_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job not found."}), 404
    return jsonify({"ok": True, "job": job})


@app.route('/api/artifacts')
def get_artifacts():
    return jsonify({"ok": True, "artifacts": _build_artifacts_payload()})


@app.route('/api/violations-summary')
def get_violations_summary():
    if not VIOLATIONS_FILE.exists():
        return jsonify({"ok": True, "summary": _build_summary_payload([])})

    try:
        with open(VIOLATIONS_FILE, 'r', encoding='utf-8') as f:
            violations = json.load(f)
    except Exception:
        violations = []

    return jsonify({"ok": True, "summary": _build_summary_payload(violations)})


@app.route('/api/code-analysis', methods=['POST'])
def code_analysis():
    payload = request.get_json(silent=True) or {}
    code_text = payload.get("code", "")
    if not isinstance(code_text, str) or not code_text.strip():
        return jsonify({"ok": False, "error": "code field is required"}), 400

    warnings = []
    plagiarism_score = 0.0

    with tempfile.TemporaryDirectory(prefix="code_analysis_", dir=str(PROJECT_ROOT)) as tmp_dir:
        submission_file = Path(tmp_dir) / "submission.py"
        submission_file.write_text(code_text, encoding="utf-8")

        try:
            score, warn = _run_copydetect(submission_file, tmp_dir)
            if score is None:
                plagiarism_score = _fallback_similarity_percent(code_text)
            else:
                plagiarism_score = float(score)
            if warn:
                warnings.append(warn)
        except Exception as exc:
            plagiarism_score = _fallback_similarity_percent(code_text)
            warnings.append(f"copydetect failed: {exc}; used fallback scoring")

    ai_probability, ai_warn = _ai_probability_percent(code_text)
    if ai_warn:
        warnings.append(ai_warn)

    response = {
        "ok": True,
        "plagiarism_score": round(plagiarism_score, 2),
        "ai_probability": round(ai_probability, 2),
        "risk_level": _risk_level(plagiarism_score, ai_probability),
        "warnings": warnings,
    }
    return jsonify(response)


@app.route('/download/report/<path:filename>')
def download_report(filename):
    try:
        target_path = (REPORTS_DIR / filename).resolve()
        target_path.relative_to(REPORTS_DIR.resolve())
    except Exception:
        return jsonify({"ok": False, "error": "Invalid report path."}), 400

    if not target_path.exists():
        return jsonify({"ok": False, "error": "Report not found."}), 404

    return send_from_directory(str(REPORTS_DIR), filename, as_attachment=True)


@app.route('/artifacts/images/<path:filename>')
def artifact_image(filename):
    try:
        target_path = (IMAGES_DIR / filename).resolve()
        target_path.relative_to(IMAGES_DIR.resolve())
    except Exception:
        return jsonify({"ok": False, "error": "Invalid image path."}), 400

    if not target_path.exists():
        return jsonify({"ok": False, "error": "Image not found."}), 404

    return send_from_directory(str(IMAGES_DIR), filename)


if __name__ == '__main__':
    app.run(debug=True)
