# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use for CPU utilizatio, comment out for GPU
# from deepface import DeepFace
import cv2
from db_connection import database_connection
from functions import recognize_faces

# Connect to the database
con = database_connection()
if con is None:
    exit()

cursor = con.cursor()

#-------------------------
# Clear tables (for testing)
# cursor.execute("DELETE FROM embedding")
# cursor.execute("TRUNCATE current_enrolled_students")
# con.commit()
#-------------------------

# video capture
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Unavailable video source")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    face_results = recognize_faces(frame, cursor)

    for first_name, img_path, distance, bbox in face_results:
        x, y, w, h = bbox
        if first_name and distance:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{first_name} (Path: {img_path}, Dist: {distance:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    # press 'q' to exit/close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
con.close()