from ultralytics import YOLO
import cv2
import os
from flask import Flask, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from datetime import datetime, timedelta
import time
import logging
import paho.mqtt.client as mqtt

# Inisialisasi Flask dan konfigurasi database
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql://postgres:admin123@localhost:5432/postgres"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Inisialisasi database dan Marshmallow
db = SQLAlchemy(app)
ma = Marshmallow(app)

# Definisi model
define_output_folder = "annotations"


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    detected_objects = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)  # Menyimpan gambar sebagai BLOB


# Load model YOLO
model_path = "best.pt"  # Path model YOLO Anda
model = YOLO(model_path)

# MQTT Configuration
mqtt_broker = "192.168.1.9"  # Alamat broker MQTT
mqtt_port = 1883
mqtt_topic = "servo/control"
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set("user2", "admin123")  # Username dan password

# Logging untuk debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


# Callback MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Terhubung ke broker MQTT.")
    else:
        logger.error(f"Gagal terhubung ke broker MQTT. Kode: {rc}")


mqtt_client.on_connect = on_connect
mqtt_client.connect(mqtt_broker, mqtt_port, 60)
mqtt_client.loop_start()


# Fungsi deteksi dan streaming video
def detect_objects_stream():
    camera_ip = "http://192.168.1.8:81/stream"  # URL stream video kamera ESP32-S3 Anda
    cap = cv2.VideoCapture(camera_ip)

    if not cap.isOpened():
        logger.error("Gagal membuka kamera. Periksa koneksi dan konfigurasi!")
        return

    frame_counter = 0
    interval = 5  # Deteksi setiap 5 frame
    detections_to_save = []
    last_detection_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = datetime.now()
        if (
            last_detection_time is None
            or current_time - last_detection_time >= timedelta(minutes=2)
        ):
            if frame_counter % interval == 0:
                frame_resized = cv2.resize(frame, (160, 120))
                results = model(frame_resized)
                detected_objects = []

                for box in results[0].boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    if class_name in ["mobil", "motor"]:
                        detected_objects.append(class_name)

                if detected_objects:
                    last_detection_time = current_time

                    annotated_frame = results[0].plot()

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"{define_output_folder}/detection_{timestamp}.jpg"
                    cv2.imwrite(image_path, annotated_frame)

                    _, buffer = cv2.imencode(".jpg", annotated_frame)
                    image_binary = buffer.tobytes()

                    detection = Detection(
                        timestamp=datetime.utcnow(),
                        detected_objects=", ".join(detected_objects),
                        image_path=image_path,
                        data=image_binary,
                    )
                    detections_to_save.append(detection)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        cv2.imshow("Detected Objects", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Streaming dihentikan oleh pengguna.")
            break

        frame_counter += 1

    if detections_to_save:
        try:
            with app.app_context():
                db.session.add_all(detections_to_save)
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving to database: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    detect_objects_stream()

