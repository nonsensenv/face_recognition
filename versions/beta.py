# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use for CPU utilizatio, comment out for GPU
# from deepface import DeepFace
import cv2
import time
from db_connection import database_connection
from functions import recognize_faces

# Connect to the database
con = database_connection()
if con is None:
    exit()

cursor = con.cursor()

#-------------------------
# Clear tables (for testing)
# cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
# cursor.execute("DELETE FROM embedding")
# cursor.execute("TRUNCATE current_enrolled_students")
# cursor.execute("TRUNCATE active_faculty")
# cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
# con.commit()
#-------------------------

# video capture
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Unavailable video source")
    exit()

# Commit changes periodically to release locks
commit_interval = 10  # seconds
last_commit_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame = cv2.resize(frame, (640, 360))
    face_results = recognize_faces(frame, cursor, con)

    for first_name, img_path, distance, bbox in face_results:
        x, y, w, h = bbox
        if first_name and distance:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{first_name} Dist: {distance:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Commit changes periodically
    current_time = time.time()
    if current_time - last_commit_time >= commit_interval:
        con.commit()
        last_commit_time = current_time

    cv2.imshow("Real-Time Face Recognition", frame)

    # press 'q' to exit/close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final commit before closing
con.commit()
cap.release()
cv2.destroyAllWindows()
cursor.close()
con.close()