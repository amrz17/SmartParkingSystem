import os
from ultralytics import YOLO
import cv2
from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from datetime import datetime

# Inisialisasi SQLAlchemy dan Marshmallow
db = SQLAlchemy()
ma = Marshmallow()


def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    # Konfigurasi koneksi PostgreSQL
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "postgresql://postgres:admin123@localhost:5432/postgres"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Inisialisasi database dan Marshmallow
    db.init_app(app)
    ma.init_app(app)

    # Definisi model
    class Detection(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
        detected_objects = db.Column(db.Text, nullable=False)
        image_path = db.Column(db.String(255), nullable=False)
        data = db.Column(
            db.LargeBinary, nullable=False
        )  # Menyimpan gambar sebagai BLOB

    # Membuat tabel di database jika belum ada
    with app.app_context():
        db.create_all()

    @app.route("/", methods=["GET"])
    def detect_objects_stream():
        model_path = "best.pt"  # Ganti dengan path model Anda
        model = YOLO(model_path)
        camera_ip = "http://192.168.1.7:81/stream"  # Ganti dengan URL stream video kamera ESP32-S3 Anda

        cap = cv2.VideoCapture(camera_ip)

        if not cap.isOpened():
            return jsonify({"error": "Gagal membuka kamera!"}), 500

        def generate_frames():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Berhenti jika frame tidak bisa dibaca

                # Ubah resolusi menjadi 160x120
                frame_resized = cv2.resize(frame, (160, 120))

                # Jalankan deteksi menggunakan model YOLO pada frame dengan resolusi rendah
                results = model(frame_resized)
                detected_objects = []

                # Iterasi hasil deteksi
                for box in results[0].boxes:
                    class_id = int(box.cls)  # Ambil indeks kelas
                    class_name = model.names[
                        class_id
                    ]  # Ambil nama kelas berdasarkan indeks
                    if class_name in [
                        "mobil",
                        "motor",
                    ]:  # Filter hanya mobil atau motor
                        detected_objects.append(class_name)

                if detected_objects:  # Jika ada mobil/motor terdeteksi
                    # Visualisasi hasil deteksi
                    annotated_frame = results[0].plot()

                    # Simpan gambar anotasi
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"annotations/detection_{timestamp}.jpg"
                    os.makedirs("annotations", exist_ok=True)
                    cv2.imwrite(image_path, annotated_frame)

                    # Konversi gambar anotasi ke format binary
                    _, buffer = cv2.imencode(".jpg", annotated_frame)
                    image_binary = buffer.tobytes()

                    # Simpan ke database menggunakan SQLAlchemy
                    detection = Detection(
                        timestamp=datetime.utcnow(),
                        detected_objects=", ".join(detected_objects),
                        image_path=image_path,
                        data=image_binary,  # Simpan gambar sebagai BLOB
                    )
                    db.session.add(detection)
                    db.session.commit()

                # Encode frame asli (sebelum resize) ke format JPEG untuk streaming
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()

                # Kirim frame sebagai streaming video
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

            cap.release()

        return (
            generate_frames(),
            200,
            {"Content-Type": "multipart/x-mixed-replace; boundary=frame"},
        )

    return app


# Entry point aplikasi
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
