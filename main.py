import cv2
import easyocr
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("📷 Camera started... Waiting for number plate")

detected_numbers = set() 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            text = reader.readtext(gray, detail=0)

            if text:
                plate_text = text[0].strip()

                if plate_text not in detected_numbers:
                    detected_numbers.add(plate_text)
                    print("🚘 Detected Number Plate:", plate_text)

