import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton,
                             QMessageBox, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from deepface import DeepFace
import numpy as np
import cv2
import mysql.connector
from db_connection import database_connection, close_database_connection

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.connect_to_database()
        self.camera_active = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.image_path = None  # File path (set only on submit)
        self.embedding = None   # Embedding (set only on submit)
        self.captured_frame = None  # Store frame in memory
        self.start_camera()  # Automatically start the camera

    def initUI(self):
        self.setWindowTitle("Face Recognition Data Entry")
        self.setGeometry(100, 100, 600, 600)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Camera Feed
        self.image_label = QLabel("Camera feed will appear here")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(500, 300)
        layout.addWidget(self.image_label)

        # Camera Control Buttons
        camera_layout = QHBoxLayout()
        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)
        camera_layout.addWidget(self.capture_button)
        layout.addLayout(camera_layout)

        # First Name
        first_name_layout = QHBoxLayout()
        first_name_layout.addWidget(QLabel("First Name:"))
        self.name_input = QLineEdit()
        first_name_layout.addWidget(self.name_input)
        layout.addLayout(first_name_layout)

        # Last Name
        last_name_layout = QHBoxLayout()
        last_name_layout.addWidget(QLabel("Last Name:"))
        self.last_name_input = QLineEdit()
        last_name_layout.addWidget(self.last_name_input)
        layout.addLayout(last_name_layout)

        # Validation Status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Validation Status:"))
        self.not_validated_radio = QRadioButton("Not Validated")
        self.validated_radio = QRadioButton("Validated")
        status_layout.addWidget(self.not_validated_radio)
        status_layout.addWidget(self.validated_radio)
        layout.addLayout(status_layout)

        # Ensure only one radio button can be selected at a time for validation status
        validation_group = QButtonGroup(self)
        validation_group.addButton(self.not_validated_radio)
        validation_group.addButton(self.validated_radio)
        self.not_validated_radio.setChecked(True)  # Set "Not Validated" as the default selected status

        # Student ID
        student_id_layout = QHBoxLayout()
        student_id_layout.addWidget(QLabel("Student ID:"))
        self.student_id_input = QLineEdit()
        student_id_layout.addWidget(self.student_id_input)
        layout.addLayout(student_id_layout)

        # Department
        dept_layout = QHBoxLayout()
        dept_layout.addWidget(QLabel("Department:"))
        self.dept_input = QLineEdit()
        dept_layout.addWidget(self.dept_input)
        layout.addLayout(dept_layout)

        # Role
        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("Role:"))
        self.none_radio = QRadioButton("None")
        self.student_radio = QRadioButton("Student")
        self.faculty_radio = QRadioButton("Faculty")
        role_layout.addWidget(self.none_radio)
        role_layout.addWidget(self.student_radio)
        role_layout.addWidget(self.faculty_radio)
        layout.addLayout(role_layout)

        # Ensure only one radio button can be selected at a time for role
        role_group = QButtonGroup(self)
        role_group.addButton(self.none_radio)
        role_group.addButton(self.student_radio)
        role_group.addButton(self.faculty_radio)
        self.none_radio.setChecked(True)  # Set "None" as the default selected role

        # Submit Button
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit_data)
        layout.addWidget(submit_button)

        # Status Label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def connect_to_database(self):
        self.conn = database_connection()
        if self.conn is None:
            QMessageBox.critical(self, "Database Error", "Could not connect to the database!")
            sys.exit(1)
        self.cursor = self.conn.cursor()

    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(2)  # Use 0 for default camera
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Camera Error", "Could not open camera!")
                return
            self.camera_active = True
            self.capture_button.setEnabled(True)
            self.timer.start(30)  # Update every 30 ms
        else:
            self.timer.stop()
            self.cap.release()
            self.camera_active = False
            self.capture_button.setEnabled(False)
            self.image_label.setText("Camera feed stopped")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to QImage
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            # Scale and display
            scaled_image = image.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(QPixmap.fromImage(scaled_image))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            # Store the frame in memory instead of saving to disk
            self.captured_frame = frame
            self.image_label.setText("Image captured (not saved yet)")
            # Clear previous path and embedding since we haven't saved yet
            self.image_path = None
            self.embedding = None

    def submit_data(self):
        if self.captured_frame is None:
            QMessageBox.warning(self, "Warning", "Please capture an image first!")
            return

        first_name = self.name_input.text().strip()
        last_name = self.last_name_input.text().strip()
        validation_status = "validated" if self.validated_radio.isChecked() else "not validated"
        student_id = self.student_id_input.text().strip()
        department = self.dept_input.text().strip()
        role = "student" if self.student_radio.isChecked() else "faculty" if self.faculty_radio.isChecked() else "none"

        if not all([first_name, student_id, department, role]):
            QMessageBox.warning(self, "Warning", "Please fill all required fields!")
            return

        try:
            # Save the image to disk only now
            if not os.path.exists("dataset_img"):
                os.makedirs("dataset_img")
            self.image_path = os.path.join("dataset_img", f"{first_name}_{last_name}.jpg")
            cv2.imwrite(self.image_path, self.captured_frame)

            # Generate embedding from the saved image
            objs = DeepFace.represent(img_path=self.image_path, model_name="ArcFace")
            embedding_list = objs[0]["embedding"]  # Store the raw 512-element list
            self.embedding = np.array(embedding_list).tobytes()  # Convert to bytes for current_enrolled_students table

            insert_stmt = """
                INSERT INTO current_enrolled_students (createdAt, img_path, first_name, last_name, validation_status, idSU, department, embedding, role)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
            """
            insert_args = (self.image_path, first_name, last_name, validation_status, student_id, department, self.embedding, role)

            self.cursor.execute(insert_stmt, insert_args)
            self.conn.commit()

            # Get the ID of the newly inserted row
            face_id = self.cursor.lastrowid

            # Insert each embedding into the embedding table (use the 512-element list)
            for idy, value in enumerate(embedding_list):  # Iterate over the raw embedding list, not the bytes
                insert_embedding_stmt = "INSERT INTO embedding (FACE_ID, DIM_NUM, VALUE) VALUES (%s, %s, %s)"
                insert_embedding_args = (face_id, idy, value)
                self.cursor.execute(insert_embedding_stmt, insert_embedding_args)
            self.conn.commit()

            # Clear fields and prepare for next submission
            self.name_input.clear()
            self.last_name_input.clear()
            self.not_validated_radio.setChecked(True)
            self.student_id_input.clear()
            self.dept_input.clear()
            self.none_radio.setChecked(True)
            self.image_label.setText("Camera feed will appear here")
            self.captured_frame = None  # Reset captured frame
            self.image_path = None      # Reset path
            self.embedding = None       # Reset embedding
            self.status_label.setText("Data inserted successfully! Ready for next submission.")
            QMessageBox.information(self, "Success", "Data inserted successfully! Ready for next submission.")

        except mysql.connector.Error as err:
            QMessageBox.critical(self, "Database Error", f"Error inserting data: {err}")
            self.conn.rollback()
            if self.image_path and os.path.exists(self.image_path):
                os.remove(self.image_path)  # Clean up if save occurred but DB failed
            self.image_path = None
            self.embedding = None
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to process image: {e}")
            if self.image_path and os.path.exists(self.image_path):
                os.remove(self.image_path)  # Clean up on processing failure
            self.image_path = None
            self.embedding = None

    def closeEvent(self, event):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        close_database_connection(self.cursor, self.conn)
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec())