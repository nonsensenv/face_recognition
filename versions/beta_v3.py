import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use for CPU utilizatio, comment out for GPU
from deepface import DeepFace
import mysql.connector
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Connect to the database
try:
    con = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="eve",
        password="137982",
        database="dreamteam"
    )
    if con.is_connected():
        print("Successfully connected to the database")
except mysql.connector.Error as err:
    print(f"Error: {err}")

cursor = con.cursor()
# Clear tables (for testing)
# cursor.execute("DELETE FROM embedding")
# cursor.execute("TRUNCATE current_enrolled_students")
# con.commit()

# 2. Populate instances (pre-existing database population)
# instances = []
# for dirpath, dirnames, filenames in os.walk("dataset_img"):
#     for filename in filenames:
#         img_path = f"{dirpath}/{filename}"
#         if ".jpg" not in img_path:
#             continue
#         objs = DeepFace.represent(img_path=img_path, model_name="ArcFace")
#         for obj in objs:
#             embedding = obj["embedding"]
#             instances.append((img_path, embedding))
    
#     # print(len(instances[0])) # --ArcFace model dimensional vector embeddings--

# # 3. Insert data with duplicate check
# for idx, instance in enumerate(instances):
#     img_path = instance[0]
#     embeddings = instance[1]
    
#     cursor.execute("SELECT ID FROM current_enrolled_students WHERE IMG_PATH = %s", (img_path,))
#     result = cursor.fetchone()
    
#     if result is None:  # Insert only if not found
#         insert_identity_stmt = "INSERT INTO current_enrolled_students (img_path, embedding, first_name, last_name, validation_status, department, idSU, role) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
#         insert_identity_args = (img_path, np.array(embeddings).tobytes(), "", "", "", "", "", "")
#         cursor.execute(insert_identity_stmt, insert_identity_args)
        
#         # Get the last inserted ID
#         face_id = cursor.lastrowid
        
#         for idy, embedding in enumerate(embeddings):
#             insert_embedding_stmt = "INSERT INTO embedding (FACE_ID, DIM_NUM, VALUE) VALUES (%s, %s, %s)"
#             insert_embedding_args = (face_id, idy, embedding)
#             cursor.execute(insert_embedding_stmt, insert_embedding_args)
# con.commit()

# 4. Real-time face recognition function with bounding box
def recognize_faces(frame, cursor, threshold=4.15):
    try:
        # Detect faces in the frame
        detected_faces = DeepFace.extract_faces(img_path=frame, detector_backend='opencv')
        
        if not detected_faces:  # No faces detected
            return []
        
        results = []
        for face in detected_faces:
            face_region = face["facial_area"]  # x, y, w, h
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            
            # Extract embedding from the detected face region
            face_img = frame[y:y+h, x:x+w]  # Crop the face
            objs = DeepFace.represent(img_path=face_img, model_name="ArcFace", enforce_detection=False)
            embedding = objs[0]["embedding"]

            target_stmts = []
            for idx, value in enumerate(embedding):
                target_stmt = f"SELECT {idx} AS dim_num, {value} AS value"
                target_stmts.append(target_stmt)
            target_stmt_final = " UNION ALL ".join(target_stmts)

            select_stmt = f"""
                SELECT img_path, distance
                FROM (
                    SELECT img_path, SQRT(SUM(value)) AS distance
                    FROM (
                        SELECT img_path, SUM((source - target) * (source - target)) AS value
                        FROM (
                            SELECT current_enrolled_students.img_path, embedding.value AS source, target.value AS target
                            FROM current_enrolled_students 
                            LEFT JOIN embedding ON current_enrolled_students.ID = embedding.FACE_ID
                            LEFT JOIN ({target_stmt_final}) AS target ON embedding.DIM_NUM = target.dim_num
                        ) AS t1
                        GROUP BY img_path
                    ) AS t2
                    GROUP BY img_path
                ) AS t3
                WHERE distance < {threshold}
                ORDER BY distance ASC
            """
            cursor.execute(select_stmt)
            query_results = cursor.fetchall()

            if query_results:
                img_path, distance = sorted(query_results, key=lambda x: x[1])[0]
                
                # Fetch the first_name corresponding to the img_path
                cursor.execute("SELECT first_name FROM current_enrolled_students WHERE img_path = %s", (img_path,))
                name_result = cursor.fetchone()
                first_name = name_result[0] if name_result and name_result[0] else "No Name"
                
                results.append((first_name, img_path, distance, (x, y, w, h)))  # Append result with bounding box coordinates
            else:
                results.append((None, None, None, (x, y, w, h)))  # Append bounding box even if no match

        return results
    except Exception as e:
        print(f"Error in recognition: {e}")
        return []

# 5. Real-time video capture and recognition with bounding box
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Unavailable video source")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to 600x600
    frame = cv2.resize(frame, (600, 600))

    # Recognize faces and get bounding boxes
    face_results = recognize_faces(frame, cursor)

    # Draw bounding boxes and labels
    for first_name, img_path, distance, bbox in face_results:
        x, y, w, h = bbox
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display recognition result
        if first_name and distance:
            label = f"{first_name} (Path: {img_path}, Dist: {distance:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Cleanup
cap.release()
cv2.destroyAllWindows()
cursor.close()
con.close()