from ultralytics import YOLO
import cv2
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from datetime import datetime
import time
import logging
import paho.mqtt.client as mqtt  # Import paho-mqtt

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
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    detected_objects = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)  # Menyimpan gambar sebagai BLOB


# Load model yang telah ditraining (pastikan path model sesuai dengan file Anda)
model_path = "best.pt"  # Ganti dengan path model Anda
model = YOLO(model_path)

# Ganti alamat IP dan port sesuai dengan kamera ESP32-S3 Anda
camera_ip = (
    "http://192.168.1.8:81/stream"  # Ganti dengan URL stream video kamera ESP32-S3 Anda
)

# Buat folder untuk menyimpan gambar yang terdeteksi
output_folder = "detected_images"

# Konfigurasi Mosquitto (sesuaikan dengan topik dan broker Anda)
mqtt_broker = "192.168.1.9"  # Alamat broker MQTT
mqtt_port = 1883  # Port broker MQTT
mqtt_topic = "parking/control"  # Topik untuk kontrol servo


# Logging untuk debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Inisialisasi MQTT client
mqtt_client = mqtt.Client()

# Menambahkan otentikasi username dan password untuk MQTT
mqtt_client.username_pw_set("user2", "admin123")  # Username dan password


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Terhubung ke broker MQTT.")
    else:
        logger.error(f"Gagal terhubung ke broker MQTT. Kode: {rc}")


mqtt_client.on_connect = on_connect
mqtt_client.connect(mqtt_broker, mqtt_port, 60)
mqtt_client.loop_start()  # Jalankan loop MQTT di background

# Buka streaming video dari kamera ESP32-S3
cap = cv2.VideoCapture(camera_ip)

if not cap.isOpened():
    logger.error(
        "Gagal membuka stream kamera! Pastikan URL kamera benar dan perangkat terhubung."
    )
    exit()

with app.app_context():  # Pastikan ada konteks Flask untuk database
    db.create_all()  # Buat tabel jika belum ada

    last_saved_time = 0  # Waktu terakhir gambar disimpan
    save_interval = 120  # Interval penyimpanan dalam detik

    try:
        while True:
            # Baca frame dari stream kamera
            ret, frame = cap.read()

            if not ret:
                logger.error("Gagal membaca frame dari kamera")
                break

            # Jalankan deteksi menggunakan model YOLO
            results = model(frame)

            # Cek apakah ada hasil deteksi
            if len(results) > 0:
                detected_classes = results[
                    0
                ].boxes.cls.tolist()  # Daftar kelas yang terdeteksi
                class_labels = results[0].names  # Nama kelas yang terdeteksi

                # Tentukan kelas motor dan mobil (misalnya 2 untuk mobil, 3 untuk motor jika menggunakan COCO dataset)
                motor_class_id = 0  # ID kelas motor (sesuaikan dengan kelas model Anda)
                car_class_id = 1  # ID kelas mobil (sesuaikan dengan kelas model Anda)

                for class_id in detected_classes:
                    if class_id == motor_class_id:
                        logger.info("Motor terdeteksi! Menyimpan gambar...")

                        # Simpan gambar hanya jika waktu interval terpenuhi
                        if time.time() - last_saved_time > save_interval:
                            img_name = (
                                f"{output_folder}/detected_{cv2.getTickCount()}.jpg"
                            )
                            cv2.imwrite(img_name, frame)

                            # Simpan ke database
                            detection = Detection(
                                detected_objects=", ".join(
                                    [class_labels[int(cls)] for cls in detected_classes]
                                ),
                                image_path=img_name,
                                data=cv2.imencode(".jpg", frame)[1].tobytes(),
                            )
                            db.session.add(detection)
                            db.session.commit()

                            last_saved_time = time.time()  # Update waktu terakhir

                        # Kirim perintah ke MQTT untuk menggerakkan servo untuk motor
                        servo_command = "open_entry_car"
                        mqtt_client.publish(mqtt_topic, servo_command)
                        logger.info(
                            f"Perintah '{servo_command}' dikirim ke broker MQTT untuk motor."
                        )

                        break  # Hentikan loop setelah perintah dikirim

                    elif class_id == car_class_id:
                        logger.info("Mobil terdeteksi! Menyimpan gambar...")

                        # Simpan gambar hanya jika waktu interval terpenuhi
                        if time.time() - last_saved_time > save_interval:
                            img_name = (
                                f"{output_folder}/detected_{cv2.getTickCount()}.jpg"
                            )
                            cv2.imwrite(img_name, frame)

                            # Simpan ke database
                            detection = Detection(
                                detected_objects=", ".join(
                                    [class_labels[int(cls)] for cls in detected_classes]
                                ),
                                image_path=img_name,
                                data=cv2.imencode(".jpg", frame)[1].tobytes(),
                            )
                            db.session.add(detection)
                            db.session.commit()

                            last_saved_time = time.time()  # Update waktu terakhir

                        # Kirim perintah ke MQTT untuk menggerakkan servo untuk mobil
                        servo_command = "open_entry_bike"
                        mqtt_client.publish(mqtt_topic, servo_command)
                        logger.info(
                            f"Perintah '{servo_command}' dikirim ke broker MQTT untuk mobil."
                        )

                        break  # Hentikan loop setelah perintah dikirim

                # Visualisasi hasil deteksi
                annotated_frame = results[0].plot()

                # Tampilkan frame yang sudah dianotasi
                cv2.imshow("Detected Objects", annotated_frame)

            # Tekan 'q' untuk keluar dari loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Keluar dari aplikasi.")
                break

    except Exception as e:
        logger.error(f"Terjadi kesalahan: {e}")
    finally:
        mqtt_client.loop_stop()  # Hentikan loop MQTT
        mqtt_client.disconnect()  # Putuskan koneksi MQTT
        cap.release()
        cv2.destroyAllWindows()
