import cv2
from ultralytics import YOLO
import subprocess
import threading

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

dog_present = False

def play_sound():
    subprocess.run(["afplay", "puppeh.wav"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    dog_detected_this_frame = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label == "dog":
                dog_detected_this_frame = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                text = f"{label} {conf:.2f}"

                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

    if dog_detected_this_frame and not dog_present:
        threading.Thread(target=play_sound, daemon=True).start()

    dog_present = dog_detected_this_frame

    cv2.imshow("Dog Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()