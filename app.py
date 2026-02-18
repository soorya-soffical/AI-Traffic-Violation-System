from flask import Flask, render_template, Response, request, jsonify, send_file
import threading
import os
from datetime import datetime
from openpyxl import Workbook

from infer_extended import run_detection, generate_frames

app = Flask(__name__)

# -----------------------------
# GLOBAL STATE
# -----------------------------
violations = []
stop_event = threading.Event()
detection_thread = None


# -----------------------------
# CALLBACK FROM YOLO
# -----------------------------
def handle_violation(v):
    v.setdefault("confidence", 1.0)
    v.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    violations.append(v)


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def index():
    videos = []
    if os.path.exists("data"):
        videos = [v for v in os.listdir("data") if v.endswith(".mp4")]
    return render_template("index_live.html", videos=videos)


# -----------------------------
# START DETECTION
# -----------------------------
@app.route("/start", methods=["POST"])
def start_detection():
    global stop_event, detection_thread, violations

    data = request.get_json()
    video_name = data.get("video")

    if not video_name:
        return jsonify({"error": "No video selected"}), 400

    video_path = os.path.join("data", video_name)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 400

    # STOP OLD DETECTION
    stop_event.set()
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=1)

    # RESET
    stop_event = threading.Event()
    violations.clear()

    # START NEW DETECTION
    detection_thread = threading.Thread(
        target=run_detection,
        args=(video_path, handle_violation, stop_event),
        daemon=True
    )
    detection_thread.start()

    return jsonify({"status": "started"})


# -----------------------------
# VIDEO STREAM (NO BLOCKING)
# -----------------------------
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------
# VIOLATIONS API
# -----------------------------
@app.route("/violations")
def get_violations():
    return jsonify(violations)


# -----------------------------
# DOWNLOAD EXCEL
# -----------------------------
@app.route("/download_excel")
def download_excel():
    if not violations:
        return "No violations to download", 400

    wb = Workbook()
    ws = wb.active
    ws.title = "Violations"
    ws.append(["Type", "Details", "Confidence", "Timestamp"])

    for v in violations:
        ws.append([
            v["type"],
            v["details"],
            v["confidence"],
            v["timestamp"]
        ])

    os.makedirs("static", exist_ok=True)
    filename = "violation_report.xlsx"
    path = os.path.join("static", filename)
    wb.save(path)

    return send_file(path, as_attachment=True)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
