import sys
import os
import time
import cv2
import numpy as np
import threading
from queue import Queue
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QMessageBox, QRadioButton, QButtonGroup, QFileDialog, QTextEdit)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from deepface import DeepFace
from db_connection import database_connection, close_database_connection, create_tables
import mysql.connector
from functions import recognize_faces

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.connect_to_database()
        self.camera_active = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.image_path = None
        self.embedding = None
        self.captured_frame = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_log_time = 0
        # Initialize threading components
        self.frame_queue = Queue(maxsize=1)  # Limit queue size to avoid backlog
        self.result_queue = Queue()  # Store recognition results
        self.processing_thread = None
        self.stop_thread = threading.Event()  # Signal to stop the thread
        self.last_processed_time = 0  # Track last processing time
        self.start_camera()
        self.start_processing_thread()

    def initUI(self):
        # Same as original, no changes needed here
        self.setWindowTitle("Face Recognition Application")
        self.setGeometry(100, 100, 1300, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        main_ui_layout = QVBoxLayout()
        main_layout.addLayout(main_ui_layout)

        camera_and_captured_layout = QHBoxLayout()
        self.image_label = QLabel("Camera feed will appear here")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(700, 500)
        camera_and_captured_layout.addWidget(self.image_label)

        self.captured_image_label = QLabel("Captured image will appear here")
        self.captured_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.captured_image_label.setFixedSize(700, 500)
        camera_and_captured_layout.addWidget(self.captured_image_label)

        main_ui_layout.addLayout(camera_and_captured_layout)

        camera_layout = QHBoxLayout()
        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)
        camera_layout.addWidget(self.capture_button)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        camera_layout.addWidget(self.upload_button)

        main_ui_layout.addLayout(camera_layout)

        first_name_layout = QHBoxLayout()
        first_name_layout.addWidget(QLabel("First Name:"))
        self.name_input = QLineEdit()
        first_name_layout.addWidget(self.name_input)
        main_ui_layout.addLayout(first_name_layout)

        last_name_layout = QHBoxLayout()
        last_name_layout.addWidget(QLabel("Last Name:"))
        self.last_name_input = QLineEdit()
        last_name_layout.addWidget(self.last_name_input)
        main_ui_layout.addLayout(last_name_layout)

        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("Role:"))
        self.none_radio = QRadioButton("None")
        self.student_radio = QRadioButton("Student")
        self.faculty_radio = QRadioButton("Faculty")
        role_layout.addWidget(self.none_radio)
        role_layout.addWidget(self.student_radio)
        role_layout.addWidget(self.faculty_radio)
        main_ui_layout.addLayout(role_layout)

        role_group = QButtonGroup(self)
        role_group.addButton(self.none_radio)
        role_group.addButton(self.student_radio)
        role_group.addButton(self.faculty_radio)
        self.none_radio.setChecked(True)

        self.student_radio.toggled.connect(self.update_id_label)
        self.faculty_radio.toggled.connect(self.update_id_label)

        self.status_layout = QHBoxLayout()
        self.status_layout.addWidget(QLabel("Validation Status:"))
        self.not_validated_radio = QRadioButton("Not Validated")
        self.validated_radio = QRadioButton("Validated")
        self.status_layout.addWidget(self.not_validated_radio)
        self.status_layout.addWidget(self.validated_radio)
        main_ui_layout.addLayout(self.status_layout)

        validation_group = QButtonGroup(self)
        validation_group.addButton(self.not_validated_radio)
        validation_group.addButton(self.validated_radio)
        self.not_validated_radio.setChecked(True)

        student_id_layout = QHBoxLayout()
        self.student_id_label = QLabel("Student ID:")
        student_id_layout.addWidget(self.student_id_label)
        self.student_id_input = QLineEdit()
        student_id_layout.addWidget(self.student_id_input)
        main_ui_layout.addLayout(student_id_layout)

        dept_layout = QHBoxLayout()
        dept_layout.addWidget(QLabel("Department:"))
        self.dept_input = QLineEdit()
        dept_layout.addWidget(self.dept_input)
        main_ui_layout.addLayout(dept_layout)

        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit_data)
        main_ui_layout.addWidget(submit_button)

        self.status_label = QLabel("Real-time face recognition is running...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_ui_layout.addWidget(self.status_label)

        log_layout = QVBoxLayout()
        log_header = QLabel("Pedestrian Entry History")
        log_header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        log_header.setStyleSheet("font-weight: bold; font-size: 16px;")
        log_layout.addWidget(log_header)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFixedHeight(150)
        log_layout.addWidget(self.log_widget)

        main_layout.addLayout(log_layout)

    def update_id_label(self):
        if self.faculty_radio.isChecked():
            self.student_id_label.setText("Faculty ID:")
            for i in range(self.status_layout.count()):
                widget = self.status_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)
        else:
            self.student_id_label.setText("Student ID:")
            for i in range(self.status_layout.count()):
                widget = self.status_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)

    def connect_to_database(self):
        self.conn = database_connection()
        if self.conn is None:
            QMessageBox.critical(self, "Database Error", "Could not connect to the database!")
            sys.exit(1)
        create_tables(self.conn)
        self.cursor = self.conn.cursor()

    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Camera Error", "Could not open camera!")
                return
            self.camera_active = True
            self.capture_button.setEnabled(True)
            self.timer.start(30)  # ~33 FPS
        else:
            self.timer.stop()
            self.cap.release()
            self.camera_active = False
            self.capture_button.setEnabled(False)
            self.image_label.setText("Camera feed stopped")

    def start_processing_thread(self):
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_thread.clear()  # Reset stop signal
            self.processing_thread = threading.Thread(target=self.process_frame, daemon=True)
            self.processing_thread.start()

    def process_frame(self):
        while not self.stop_thread.is_set():
            try:
                # Get frame from queue (non-blocking if empty)
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    results, logs = recognize_faces(frame, self.cursor, self.conn)
                    self.result_queue.put((results, logs))
                    self.frame_queue.task_done()
                else:
                    time.sleep(0.01)  # Prevent CPU overuse
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)  # Back off on errors

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Send frame for processing only every 500ms to avoid overloading
        current_time = time.time()
        if current_time - self.last_processed_time >= 0.1:
            try:
                # Add frame to queue if not full
                if self.frame_queue.qsize() == 0:
                    self.frame_queue.put(frame.copy())
                    self.last_processed_time = current_time
            except:
                pass  # Queue full, skip frame

        # Retrieve results if available
        face_results = []
        logs = []
        while not self.result_queue.empty():
            face_results, logs = self.result_queue.get()
            self.result_queue.task_done()

        # Update logs every 5 seconds
        if logs and current_time - self.last_log_time >= 5:
            for log in logs:
                print(log)
                self.log_widget.append(f"• {log}")
            self.last_log_time = current_time

        # Draw face results on the frame
        for first_name, img_path, distance, bbox in face_results:
            x, y, w, h = bbox
            if first_name and distance:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{first_name} (Dist: {distance:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.cursor.execute("""
                    SELECT validation_status 
                    FROM current_enrolled_students 
                    WHERE first_name = %s AND img_path = %s
                    UNION
                    SELECT validation_status 
                    FROM active_faculty 
                    WHERE first_name = %s AND img_path = %s
                """, (first_name, img_path, first_name, img_path))
                validation_status = self.cursor.fetchone()
                validation_status = validation_status[0] if validation_status else "not validated"

                color = (0, 255, 0) if validation_status == "validated" else (0, 0, 255)
                cv2.putText(frame, validation_status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display frame in GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_image = image.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(scaled_image))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_frame = frame
            self.image_label.setText("Image captured (not saved yet)")
            self.image_path = None
            self.embedding = None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_image = image.scaled(self.captured_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.captured_image_label.setPixmap(QPixmap.fromImage(scaled_image))

    def upload_image(self):
        print("Upload button clicked!")
        self.timer.stop()
        try:
            print("Opening file dialog...")
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an Image",
                "",
                "Images (*.png *.jpg *.jpeg)"
            )
            print(f"Selected file: {file_path}")
            if file_path:
                self.image_path = file_path
                image = cv2.imread(file_path)
                if image is not None:
                    self.captured_frame = image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    scaled_image = q_image.scaled(self.captured_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.captured_image_label.setPixmap(QPixmap.fromImage(scaled_image))
                    self.captured_image_label.setText("")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load the selected image.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading the image: {e}")
        finally:
            if self.camera_active:
                self.timer.start(30)

    def submit_data(self):
        if self.captured_frame is None:
            QMessageBox.warning(self, "Warning", "Please capture or upload an image first!")
            return

        first_name = self.name_input.text().strip()
        last_name = self.last_name_input.text().strip()
        validation_status = "validated" if self.validated_radio.isChecked() else "not validated"
        student_id = self.student_id_input.text().strip()
        department = self.dept_input.text().strip()
        role = "student" if self.student_radio.isChecked() else "faculty" if self.faculty_radio.isChecked() else "none"

        if not all([first_name, student_id, department, role]) or role == "none":
            QMessageBox.warning(self, "Warning", "Please fill all required fields and select a valid role!")
            return

        try:
            if not os.path.exists("dataset_img"):
                os.makedirs("dataset_img")
            self.image_path = os.path.join("dataset_img", f"{first_name}_{last_name}.jpg")
            cv2.imwrite(self.image_path, self.captured_frame)

            objs = DeepFace.represent(img_path=self.image_path, model_name="ArcFace", enforce_detection=True)
            embedding_list = objs[0]["embedding"]
            self.embedding = np.array(embedding_list).tobytes()

            if role == "student":
                insert_stmt = """
                    INSERT INTO current_enrolled_students (createdAt, img_path, first_name, last_name, validation_status, idSU, department, embedding, role)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                """
                insert_args = (self.image_path, first_name, last_name, validation_status, student_id, department, self.embedding, role)
                self.cursor.execute(insert_stmt, insert_args)
                self.conn.commit()
                student_id = self.cursor.lastrowid

                for idy, value in enumerate(embedding_list):
                    insert_embedding_stmt = "INSERT INTO embedding (STUDENT_ID, FACE_ID, DIM_NUM, VALUE) VALUES (%s, %s, %s, %s)"
                    insert_embedding_args = (student_id, student_id, idy, value)
                    self.cursor.execute(insert_embedding_stmt, insert_embedding_args)
            elif role == "faculty":
                insert_stmt = """
                    INSERT INTO active_faculty (createdAt, img_path, first_name, last_name, idSU, department, embedding, role)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
                """
                insert_args = (self.image_path, first_name, last_name, student_id, department, self.embedding, role)
                self.cursor.execute(insert_stmt, insert_args)
                self.conn.commit()
                faculty_id = self.cursor.lastrowid

                for idy, value in enumerate(embedding_list):
                    insert_embedding_stmt = "INSERT INTO embedding (FACULTY_ID, FACE_ID, DIM_NUM, VALUE) VALUES (%s, %s, %s, %s)"
                    insert_embedding_args = (faculty_id, faculty_id, idy, value)
                    self.cursor.execute(insert_embedding_stmt, insert_embedding_args)
            self.conn.commit()

            self.name_input.clear()
            self.last_name_input.clear()
            self.not_validated_radio.setChecked(True)
            self.student_id_input.clear()
            self.dept_input.clear()
            self.none_radio.setChecked(True)
            self.image_label.setText("Camera feed will appear here")
            self.captured_image_label.setText("Captured image will appear here")
            self.captured_frame = None
            self.image_path = None
            self.embedding = None
            self.status_label.setText("Data inserted successfully! Ready for next submission.")
            QMessageBox.information(self, "Success", "Data inserted successfully! Ready for next submission.")

        except mysql.connector.Error as err:
            QMessageBox.critical(self, "Database Error", f"Error inserting data: {err}")
            self.conn.rollback()
            if self.image_path and os.path.exists(self.image_path):
                os.remove(self.image_path)
            self.image_path = None
            self.embedding = None
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to process image: {e}")
            if self.image_path and os.path.exists(self.image_path):
                os.remove(self.image_path)
            self.image_path = None
            self.embedding = None

    def closeEvent(self, event):
        self.stop_thread.set()  # Signal thread to stop
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        close_database_connection(self.cursor, self.conn)
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())