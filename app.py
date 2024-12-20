import os
from ultralytics import YOLO
import cv2
from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from dotenv import load_dotenv
from datetime import datetime

# Muat variabel dari .env
load_dotenv()

# Inisialisasi SQLAlchemy dan Marshmallow
db = SQLAlchemy()
ma = Marshmallow()

def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    # Konfigurasi koneksi PostgreSQL
    app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:Kiic2023@localhost:5432/postgres"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Inisialisasi database dan Marshmallow
    db.init_app(app)
    ma.init_app(app)

    # Definisi model
    class Detection(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
        detected_objects = db.Column(db.Text, nullable=False)
        image_path = db.Column(db.String(255), nullable=False)

    # Membuat tabel di database jika belum ada
    with app.app_context():
        db.create_all()

    @app.route('/detect', methods=['GET'])
    def detect_objects():
        model_path = "best.pt"  # Ganti dengan path model Anda
        model = YOLO(model_path)
        cap = cv2.VideoCapture(0)  # Ganti dengan sumber video yang sesuai

        if not cap.isOpened():
            return jsonify({"error": "Gagal membuka kamera!"}), 500

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return jsonify({"error": "Gagal membaca frame dari kamera!"}), 500

        # Jalankan deteksi menggunakan model YOLO
        results = model(frame)
        detected_objects = ', '.join([res.name for res in results[0].boxes])

        # Visualisasi hasil deteksi
        annotated_frame = results[0].plot()

        # Simpan gambar anotasi
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f'annotations/detection_{timestamp}.jpg'
        os.makedirs('annotations', exist_ok=True)
        cv2.imwrite(image_path, annotated_frame)

        # Simpan ke database menggunakan SQLAlchemy
        detection = Detection(
            timestamp=datetime.utcnow(),
            detected_objects=detected_objects,
            image_path=image_path
        )
        db.session.add(detection)
        db.session.commit()

        cap.release()

        return jsonify({
            "timestamp": timestamp,
            "detected_objects": detected_objects,
            "image_path": image_path
        })

    return app


# Entry point aplikasi
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
