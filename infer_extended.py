import cv2
import time
from ultralytics import YOLO

# ============================
# LOAD MODELS
# ============================
vehicle_model =\
    YOLO("models/best.pt")      # COCO model
helmet_model = YOLO("models/helmet.pt")     # Helmet model

latest_frame = None


# ============================
# HELPER: CHECK OVERLAP
# ============================
def is_inside(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2

    return bx1 < cx < bx2 and by1 < cy < by2


# ============================
# MAIN DETECTION
# ============================
def run_detection(video_path, callback, stop_event):
    global latest_frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 / fps if fps > 0 else 0.03

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- DETECT OBJECTS ----------------
        results = vehicle_model(frame, conf=0.4, verbose=False)

        bikes = []
        persons = []

        for r in results:
            for box in r.boxes:
                cls = vehicle_model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls in ["motorcycle", "motorbike"]:
                    bikes.append((x1, y1, x2, y2))
                elif cls == "person":
                    persons.append((x1, y1, x2, y2))

        # ---------------- PROCESS EACH BIKE ----------------
        for bx1, by1, bx2, by2 in bikes:
            riders = []

            for p in persons:
                if is_inside(p, (bx1, by1, bx2, by2)):
                    riders.append(p)

            # 🚨 OVERLOADING VIOLATION
            if len(riders) > 2:
                callback({
                    "type": "overloading",
                    "details": f"{len(riders)} or more people on bike",
                    "confidence": 1.0
                })
                bike_color = (0, 0, 255)
            else:
                bike_color = (0, 255, 0)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), bike_color, 3)
            cv2.putText(
                frame,
                f"Riders: {len(riders)}",
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                bike_color,
                2
            )

            # ---------------- HELMET CHECK PER RIDER ----------------
            helmet_results = helmet_model(frame, conf=0.4, verbose=False)

            for px1, py1, px2, py2 in riders:
                helmet_found = False

                for hr in helmet_results:
                    for hb in hr.boxes:
                        hcls = helmet_model.names[int(hb.cls[0])]
                        hx1, hy1, hx2, hy2 = map(int, hb.xyxy[0])

                        if hcls == "helmet" and hx1 > px1 and hx2 < px2:
                            helmet_found = True

                if not helmet_found:
                    callback({
                        "type": "helmet_violation",
                        "details": "Rider without helmet",
                        "confidence": 1.0
                    })
                    color = (0, 0, 255)
                    label = "NO HELMET"
                else:
                    color = (0, 255, 0)
                    label = "HELMET"

                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        latest_frame = frame.copy()
        time.sleep(delay)

    cap.release()


# ============================
# VIDEO STREAM
# ============================
def generate_frames():
    global latest_frame

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        ret, buffer = cv2.imencode(".jpg", latest_frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )
