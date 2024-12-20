# Import library yang dibutuhkan
from ultralytics import YOLO
import cv2

# Load model yang telah ditraining (pastikan path model sesuai dengan file Anda)
model_path = "best.pt"  # Ganti dengan path model Anda
model = YOLO(model_path)

# Buka kamera (biasanya kamera default adalah 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Gagal membuka kamera!")
    exit()

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame dari kamera")
        break

    # Jalankan deteksi menggunakan model YOLO
    results = model(frame)

    # Visualisasi hasil deteksi
    annotated_frame = results[0].plot()

    # Tampilkan frame yang sudah dianotasi
    cv2.imshow("Detected Objects", annotated_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup kamera dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
