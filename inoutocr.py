from ultralytics import YOLO
import cv2
import easyocr
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from datetime import datetime

# Inisialisasi Flask dan konfigurasi database
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql://postgres:admin123@localhost:5432/postgres"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Inisialisasi database dan Marshmallow
db = SQLAlchemy(app)
ma = Marshmallow(app)


# Definisi model untuk database
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    detected_objects = db.Column(db.Text, nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)  # Menyimpan gambar sebagai BLOB
    plate_image = db.Column(db.LargeBinary, nullable=False)  # Gambar plat nomor
    plate_number = db.Column(db.String(20), nullable=False)  # Teks plat nomor
    confidence_plate = db.Column(db.String(255), nullable=True)  # Tingkat keyakinan OCR
    start_parking = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    finish_parking = db.Column(db.DateTime, nullable=True)
    total_time = db.Column(db.Integer, nullable=True)


# Fungsi Pre-Processing untuk OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.GaussianBlur(binary, (5, 5), 0)
    resized = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    return padded


# Load model YOLO
model_path = "best.pt"  # Ganti dengan path model Anda
model = YOLO(model_path)

# Inisialisasi EasyOCR
reader = easyocr.Reader(["en", "id"])

# Ganti alamat IP dan port sesuai dengan kamera ESP32-S3 Anda
camera_ip = (
    "http://192.168.1.7:81/stream"  # Ganti dengan URL stream video kamera ESP32-S3 Anda
)

# Buka streaming video dari kamera pertama (masuk) dan kamera kedua (keluar)
cam1 = cv2.VideoCapture(camera_ip)  # Kamera untuk mendeteksi kendaraan keluar
cam2 = cv2.VideoCapture(3)  # Kamera untuk mendeteksi kendaraan masuk

if not cam1.isOpened() or not cam2.isOpened():
    print("Gagal membuka salah satu stream kamera! Pastikan kamera terhubung.")
    exit()


# Fungsi untuk mendeteksi kendaraan di kamera
def detect_vehicle(camera, context):
    ret, frame = camera.read()

    if not ret:
        print(f"Gagal membaca frame dari {context}.")
        return None, None

    # Jalankan deteksi menggunakan model YOLO
    results = model(frame)

    # OCR untuk plat nomor
    plate_text = None
    for box in results[0].boxes.xyxy:  # Bounding box koordinat
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_plate = preprocess_image(frame[y_min:y_max, x_min:x_max])
        ocr_results = reader.readtext(cropped_plate)

        if ocr_results:
            plate_text = " ".join([res[1] for res in ocr_results])
            break  # Berhenti setelah menemukan OCR pertama yang valid

    return plate_text, frame


with app.app_context():
    db.create_all()  # Buat tabel jika belum ada

    while True:
        # Deteksi kendaraan masuk
        plate_text_in, frame_in = detect_vehicle(cam1, "kamera masuk")
        active_plates = set()
        if plate_text_in:
            if plate_text_in in active_plates:
                print(f"Plat nomor {plate_text_in} sudah aktif. Tidak menyimpan ulang.")
            else:
                active_plates.add(plate_text_in)
                print(
                    f"Plat nomor {plate_text_in} masuk dan ditambahkan ke daftar aktif."
                )

                # Simpan ke database
                detection = Detection(
                    detected_objects="Masuk",  # Tandai kendaraan masuk
                    data=cv2.imencode(".jpg", frame_in)[1].tobytes(),
                    plate_image=cv2.imencode(".jpg", frame_in)[1].tobytes(),
                    plate_number=plate_text_in,
                )
                db.session.add(detection)
                db.session.commit()
        else:
            print("Data tidak valid: confidence_plate tidak terdeteksi.")

        # Deteksi kendaraan keluar
        plate_text_out, frame_out = detect_vehicle(cam2, "kamera keluar")
        if plate_text_out:
            if plate_text_out in active_plates:
                active_plates.remove(plate_text_out)
                print(
                    f"Plat nomor {plate_text_out} keluar dan dihapus dari daftar aktif."
                )

                # Simpan ke database sebagai keluar
                detection = Detection(
                    detected_objects="Keluar",  # Tandai kendaraan keluar
                    data=cv2.imencode(".jpg", frame_out)[1].tobytes(),
                    plate_image=cv2.imencode(".jpg", frame_out)[1].tobytes(),
                    plate_number=plate_text_out,
                )
                db.session.add(detection)
                db.session.commit()

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Tutup semua stream dan jendela tampilan
cam1.release()
cam2.release()
cv2.destroyAllWindows()
