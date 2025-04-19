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
from PyQt6.QtGui import QImage, QPixmap, QScreen
from deepface import DeepFace
from db_connection import database_connection, close_database_connection, create_tables
import mysql.connector
from functions import recognize_faces

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI() 
        self.center_window() 
        self.connect_to_database() 
        self.camera_active = False  
        self.timer = QTimer(self)  
        self.timer.timeout.connect(self.update_frame) 
        self.image_path = None
        self.embedding = None
        self.captured_frame = None  
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
        self.last_log_time = 0 
        # Threading setup
        self.frame_queue = Queue(maxsize=1)  
        self.result_queue = Queue() 
        self.processing_thread = None 
        self.stop_thread = threading.Event() 
        self.last_processed_time = 0  
        self.start_camera() 
        self.start_processing_thread() 

    def center_window(self):

        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def initUI(self):

        self.setWindowTitle("Face Recognition Application")
        self.setGeometry(100, 100, 1300, 600)  
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)  
        main_ui_layout = QVBoxLayout()
        main_layout.addLayout(main_ui_layout)

        # Camera and captured image display
        camera_and_captured_layout = QHBoxLayout()
        self.image_label = QLabel("Camera feed will appear here")  # Label for live feed
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(700, 500)  # Fixed size for camera feed
        camera_and_captured_layout.addWidget(self.image_label)

        self.captured_image_label = QLabel("Captured image will appear here")  # Label for captured image
        self.captured_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.captured_image_label.setFixedSize(700, 500)
        camera_and_captured_layout.addWidget(self.captured_image_label)
        main_ui_layout.addLayout(camera_and_captured_layout)

        # Camera control buttons
        camera_layout = QHBoxLayout()
        self.capture_button = QPushButton("Capture Image")  # Button to capture frame
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)  # Disable until camera starts
        camera_layout.addWidget(self.capture_button)

        self.upload_button = QPushButton("Upload Image")  # Button to upload image
        self.upload_button.clicked.connect(self.upload_image)
        camera_layout.addWidget(self.upload_button)
        main_ui_layout.addLayout(camera_layout)

        # First name input
        first_name_layout = QHBoxLayout()
        first_name_layout.addWidget(QLabel("First Name:"))
        self.name_input = QLineEdit()  # Text field for first name
        first_name_layout.addWidget(self.name_input)
        main_ui_layout.addLayout(first_name_layout)

        # Last name input
        last_name_layout = QHBoxLayout()
        last_name_layout.addWidget(QLabel("Last Name:"))
        self.last_name_input = QLineEdit()  # Text field for last name
        last_name_layout.addWidget(self.last_name_input)
        main_ui_layout.addLayout(last_name_layout)

        # Role selection
        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("Role:"))
        self.none_radio = QRadioButton("None")
        self.student_radio = QRadioButton("Student")
        self.faculty_radio = QRadioButton("Faculty")
        role_layout.addWidget(self.none_radio)
        role_layout.addWidget(self.student_radio)
        role_layout.addWidget(self.faculty_radio)
        main_ui_layout.addLayout(role_layout)

        # Group role radio buttons
        role_group = QButtonGroup(self)
        role_group.addButton(self.none_radio)
        role_group.addButton(self.student_radio)
        role_group.addButton(self.faculty_radio)
        self.none_radio.setChecked(True)  # Default to None

        # Update ID label when role changes
        self.student_radio.toggled.connect(self.update_id_label)
        self.faculty_radio.toggled.connect(self.update_id_label)

        # Validation status selection
        self.status_layout = QHBoxLayout()
        self.status_layout.addWidget(QLabel("Validation Status:"))
        self.not_validated_radio = QRadioButton("Not Validated")
        self.validated_radio = QRadioButton("Validated")
        self.status_layout.addWidget(self.not_validated_radio)
        self.status_layout.addWidget(self.validated_radio)
        main_ui_layout.addLayout(self.status_layout)

        # Group validation radio buttons
        validation_group = QButtonGroup(self)
        validation_group.addButton(self.not_validated_radio)
        validation_group.addButton(self.validated_radio)
        self.not_validated_radio.setChecked(True)  # Default to Not Validated

        # ID input
        student_id_layout = QHBoxLayout()
        self.student_id_label = QLabel("Student ID:")  # Label for ID
        student_id_layout.addWidget(self.student_id_label)
        self.student_id_input = QLineEdit()  # Text field for ID
        student_id_layout.addWidget(self.student_id_input)
        main_ui_layout.addLayout(student_id_layout)

        # Department input
        dept_layout = QHBoxLayout()
        dept_layout.addWidget(QLabel("Department:"))
        self.dept_input = QLineEdit()  # Text field for department
        dept_layout.addWidget(self.dept_input)
        main_ui_layout.addLayout(dept_layout)

        # Submit button
        submit_button = QPushButton("Submit")  # Button to save data
        submit_button.clicked.connect(self.submit_data)
        main_ui_layout.addWidget(submit_button)

        # Status message
        self.status_label = QLabel("Real-time face recognition is running...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_ui_layout.addWidget(self.status_label)

        # Log display
        log_layout = QVBoxLayout()
        log_header = QLabel("Pedestrian Entry History")
        log_header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        log_header.setStyleSheet("font-weight: bold; font-size: 16px;")  # Style log header
        log_layout.addWidget(log_header)

        self.log_widget = QTextEdit()  # Text area for logs
        self.log_widget.setReadOnly(True)  # Prevent editing
        self.log_widget.setFixedHeight(150)  # Fixed height
        log_layout.addWidget(self.log_widget)
        main_layout.addLayout(log_layout)

    def update_id_label(self):
        # Update ID label based on role
        if self.faculty_radio.isChecked():
            self.student_id_label.setText("Faculty ID:")
            # Hide validation status for faculty
            for i in range(self.status_layout.count()):
                widget = self.status_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)
        else:
            self.student_id_label.setText("Student ID:")
            # Show validation status
            for i in range(self.status_layout.count()):
                widget = self.status_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)

    def connect_to_database(self):
        # Connect to MySQL
        self.conn = database_connection()
        if self.conn is None:
            QMessageBox.critical(self, "Database Error", "Could not connect to the database!")
            sys.exit(1)  # Exit on failure
        create_tables(self.conn)  # Create database tables
        self.cursor = self.conn.cursor()  # Initialize cursor

    def start_camera(self):
        # Start or stop camera
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)  # Open default camera
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Camera Error", "Could not open camera!")
                return
            self.camera_active = True
            self.capture_button.setEnabled(True)  # Enable capture button
            self.timer.start(30)  # Update every 30ms
        else:
            self.timer.stop()  # Stop updates
            self.cap.release()  # Release camera
            self.camera_active = False
            self.capture_button.setEnabled(False)  # Disable capture button
            self.image_label.setText("Camera feed stopped")

    def start_processing_thread(self):
        # Start thread for face recognition
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_thread.clear()  # Reset stop signal
            self.processing_thread = threading.Thread(target=self.process_frame, daemon=True)  # Daemon thread
            self.processing_thread.start()  # Start thread

    def process_frame(self):
        # Process frames in separate thread
        while not self.stop_thread.is_set():
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()  # Get frame from queue
                    results, logs = recognize_faces(frame, self.cursor, self.conn)  # Recognize faces
                    self.result_queue.put((results, logs))  # Store results
                    self.frame_queue.task_done()  # Mark frame processed
                else:
                    time.sleep(0.01)  # Avoid busy loop
            except Exception as e:
                print(f"Processing error: {e}")  # Log errors
                time.sleep(0.1)  # Prevent rapid retries

    def update_frame(self):
        # Update camera feed
        ret, frame = self.cap.read()  # Read frame
        if not ret:
            return  # Skip if failed

        current_time = time.time()
        # Add frame to queue every 0.1s
        if current_time - self.last_processed_time >= 0.1:
            try:
                if self.frame_queue.qsize() == 0:
                    self.frame_queue.put(frame.copy())  # Copy to avoid overwrite
                    self.last_processed_time = current_time
            except:
                pass

        # Get recognition results
        face_results = []
        logs = []
        while not self.result_queue.empty():
            face_results, logs = self.result_queue.get()  # Retrieve results
            self.result_queue.task_done()

        # Display logs every 5s
        if logs and current_time - self.last_log_time >= 5:
            for log in logs:
                print(log)  # Print to console
                self.log_widget.append(f"• {log}")  # Add to GUI
            self.last_log_time = current_time

        # Draw face boxes and labels
        for first_name, img_path, distance, bbox in face_results:
            x, y, w, h = bbox
            if first_name and distance:
                # Draw green box for known face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{first_name} (Dist: {distance:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check validation status
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

                # Color based on status
                color = (0, 255, 0) if validation_status == "validated" else (0, 0, 255)
                cv2.putText(frame, validation_status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Draw red box for unknown face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Convert frame for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)  # Create QImage
        scaled_image = image.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)  # Scale to label
        self.image_label.setPixmap(QPixmap.fromImage(scaled_image))  # Display frame

    def capture_image(self):
        # Capture frame from camera
        ret, frame = self.cap.read()
        if ret:
            self.captured_frame = frame  # Store frame
            self.image_label.setText("Image captured (not saved yet)")
            self.image_path = None  # Clear path
            self.embedding = None  # Clear embedding

            # Display captured frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_image = image.scaled(self.captured_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.captured_image_label.setPixmap(QPixmap.fromImage(scaled_image))

    def upload_image(self):
        # Upload image from file
        print("Upload button clicked!")
        self.timer.stop()  # Pause camera
        try:
            print("Opening file dialog...")
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an Image",
                "",
                "Images (*.png *.jpg *.jpeg)"
            )  # Open file dialog
            print(f"Selected file: {file_path}")
            if file_path:
                self.image_path = file_path
                image = cv2.imread(file_path)  # Read image
                if image is not None:
                    self.captured_frame = image  # Store image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    scaled_image = q_image.scaled(self.captured_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.captured_image_label.setPixmap(QPixmap.fromImage(scaled_image))  # Display image
                    self.captured_image_label.setText("")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load the selected image.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading the image: {e}")
        finally:
            if self.camera_active:
                self.timer.start(30)  # Resume camera

    def submit_data(self):
        # Submit captured data to database
        if self.captured_frame is None:
            QMessageBox.warning(self, "Warning", "Please capture or upload an image first!")
            return

        # Get input data
        first_name = self.name_input.text().strip()
        last_name = self.last_name_input.text().strip()
        validation_status = "validated" if self.validated_radio.isChecked() else "not validated"
        student_id = self.student_id_input.text().strip()
        department = self.dept_input.text().strip()
        role = "student" if self.student_radio.isChecked() else "faculty" if self.faculty_radio.isChecked() else "none"

        # Validate inputs
        if not all([first_name, student_id, department, role]) or role == "none":
            QMessageBox.warning(self, "Warning", "Please fill all required fields and select a valid role!")
            return

        try:
            # Save image to disk
            if not os.path.exists("dataset_img"):
                os.makedirs("dataset_img")  # Create directory
            self.image_path = os.path.join("dataset_img", f"{first_name}_{last_name}.jpg")
            cv2.imwrite(self.image_path, self.captured_frame)  # Save image

            # Generate face embedding
            objs = DeepFace.represent(img_path=self.image_path, model_name="ArcFace", enforce_detection=True)
            embedding_list = objs[0]["embedding"]
            self.embedding = np.array(embedding_list).tobytes()  # Convert to bytes

            if role == "student":
                # Insert student data
                insert_stmt = """
                    INSERT INTO current_enrolled_students (createdAt, img_path, first_name, last_name, validation_status, idSU, department, embedding, role)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                """
                insert_args = (self.image_path, first_name, last_name, validation_status, student_id, department, self.embedding, role)
                self.cursor.execute(insert_stmt, insert_args)
                self.conn.commit()
                student_id = self.cursor.lastrowid  # Get inserted ID

                # Store embedding dimensions
                for idy, value in enumerate(embedding_list):
                    insert_embedding_stmt = "INSERT INTO embedding (STUDENT_ID, FACE_ID, DIM_NUM, VALUE) VALUES (%s, %s, %s, %s)"
                    insert_embedding_args = (student_id, student_id, idy, value)
                    self.cursor.execute(insert_embedding_stmt, insert_embedding_args)
            elif role == "faculty":
                # Insert faculty data
                insert_stmt = """
                    INSERT INTO active_faculty (createdAt, img_path, first_name, last_name, idSU, department, embedding, role)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
                """
                insert_args = (self.image_path, first_name, last_name, student_id, department, self.embedding, role)
                self.cursor.execute(insert_stmt, insert_args)
                self.conn.commit()
                faculty_id = self.cursor.lastrowid

                # Store embedding dimensions
                for idy, value in enumerate(embedding_list):
                    insert_embedding_stmt = "INSERT INTO embedding (FACULTY_ID, FACE_ID, DIM_NUM, VALUE) VALUES (%s, %s, %s, %s)"
                    insert_embedding_args = (faculty_id, faculty_id, idy, value)
                    self.cursor.execute(insert_embedding_stmt, insert_embedding_args)
            self.conn.commit()  # Commit changes

            # Clear inputs and reset UI
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
            self.conn.rollback()  # Undo changes
            if self.image_path and os.path.exists(self.image_path):
                os.remove(self.image_path)  # Delete saved image
            self.image_path = None
            self.embedding = None
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to process image: {e}")
            if self.image_path and os.path.exists(self.image_path):
                os.remove(self.image_path)
            self.image_path = None
            self.embedding = None

    def closeEvent(self, event):
        # Clean up on window close
        self.stop_thread.set()  # Signal thread to stop
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)  # Wait for thread
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()  # Release camera
        close_database_connection(self.cursor, self.conn)  # Close database
        event.accept()  # Accept close event

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.center_window()  # Center the window

    def center_window(self):
        # Center the window on the screen
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def initUI(self):
        self.setWindowTitle("Login")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout(self)

        # Username input
        username_layout = QHBoxLayout()
        username_layout.addWidget(QLabel("Username:"))
        self.username_input = QLineEdit()
        username_layout.addWidget(self.username_input)
        layout.addLayout(username_layout)

        # Password input
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        password_layout.addWidget(self.password_input)
        layout.addLayout(password_layout)

        # Login button
        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.authenticate)
        layout.addWidget(self.login_button)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def authenticate(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        conn = database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM admin WHERE username = %s", (username,))
            result = cursor.fetchone()
            close_database_connection(cursor, conn)

            if result and result[0] == password:
                self.status_label.setText("Login successful!")
                self.open_main_window()
            else:
                self.status_label.setText("Invalid username or password.")
        else:
            self.status_label.setText("Database connection error.")

    def open_main_window(self):
        self.main_window = FaceRecognitionApp()
        self.main_window.show()
        self.close()

if __name__ == '__main__':
    # Start application
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec())