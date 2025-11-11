from ultralytics import YOLO
import os

model = YOLO(r'C:\Users\user\runs\detect\train18\weights\best.pt')

source_dir = r'deteksi_makanan\test\images'
save_dir = r'deteksi_makanan\result'

os.makedirs(save_dir, exist_ok=True)


results = model.predict(
    source=source_dir,   
    save=True,           
    project=save_dir,    
    name='',             
    exist_ok=True,       
    imgsz=640,           
    conf=0.25            
)

print(f"Hasil prediksi semua gambar tersimpan di: {os.path.abspath(save_dir)}")