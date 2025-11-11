# Bikin Model

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data=r'deteksi_makanan\data.yaml',
    epochs=50,
    imgsz=640
)

results = model.predict(
    source=r'deteksi_makanan\test\images\IMG_20251111_130635_036_jpg.rf.c63428a9d498593513820fbc547be6b1.jpg',
    show=True
)