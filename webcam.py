from ultralytics import YOLO
import cv2

model = YOLO(r'C:\Users\user\runs\detect\train18\weights\best.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam is not connected!")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame is not recognized")
        break

    results = model(frame, imgsz=640, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("Deteksi Makanan - YOLOv8", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()