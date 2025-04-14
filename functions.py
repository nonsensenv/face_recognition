from deepface import DeepFace
import time
import os
import cv2  # Add OpenCV for saving snapshots

last_print_time = 0
last_unknown_snapshot_time = 0  # Track the last snapshot time for unknown faces

def recognize_faces(frame, cursor, con, threshold=3.6):
    global last_print_time, last_unknown_snapshot_time
    logs = []  # Collect logs to return
    try:
        detected_faces = DeepFace.extract_faces(img_path=frame, detector_backend='opencv')
        
        if not detected_faces: 
            return [], logs
        
        results = []
        for face in detected_faces:
            face_region = face["facial_area"] 
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            
            face_img = frame[y:y+h, x:x+w]
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
                            UNION ALL
                            SELECT active_faculty.img_path, embedding.value AS source, target.value AS target
                            FROM active_faculty 
                            LEFT JOIN embedding ON active_faculty.ID = embedding.FACE_ID
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
                
                cursor.execute("""
                    SELECT first_name, last_name, idSU, department, role, 
                           CASE WHEN role = 'faculty' THEN 'Validated (Faculty)' ELSE validation_status END AS validation_status
                    FROM current_enrolled_students WHERE img_path = %s
                    UNION
                    SELECT first_name, last_name, idSU, department, role, 'Validated (Faculty)' AS validation_status
                    FROM active_faculty WHERE img_path = %s
                """, (img_path, img_path))
                details = cursor.fetchone()
                if details:
                    first_name, last_name, idsu, department, role, validation_status = details
                    results.append((first_name, img_path, distance, (x, y, w, h)))
                    
                    # Fetch the current timestamp
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Updated log to include validation status and datetime
                    #logs.append(f"Detected Entry: {first_name} {last_name}, IDSU: {idsu}, Department: {department}, Role: {role}, Validation Status: {validation_status}, Datetime: {current_time}")
                
                cursor.execute("""
                    SELECT COUNT(*) FROM pedestrian_entry_history
                    WHERE (STUDENT_ID = (SELECT id FROM current_enrolled_students WHERE img_path = %s)
                    OR FACULTY_ID = (SELECT id FROM active_faculty WHERE img_path = %s))
                    AND ENTRY_TIME > NOW() - INTERVAL 5 SECOND
                """, (img_path, img_path))
                entry_exists = cursor.fetchone()[0]
                
                if entry_exists == 0:
                    cursor.execute("""
                        INSERT INTO pedestrian_entry_history (STUDENT_ID, FACULTY_ID, idsu, first_name, last_name, department, role, validation_status)
                        SELECT id, NULL, idSU, first_name, last_name, department, role, validation_status FROM current_enrolled_students WHERE img_path = %s
                        UNION
                        SELECT NULL, id, idSU, first_name, last_name, department, role, 'Validated (Faculty)' FROM active_faculty WHERE img_path = %s
                    """, (img_path, img_path))
                    con.commit()
                    
                    cursor.execute("""
                        SELECT peh.*, DATE_FORMAT(peh.ENTRY_TIME, '%y/%m/%d %H:%i:%s') as formatted_time
                        FROM pedestrian_entry_history peh
                        ORDER BY peh.ENTRY_TIME DESC LIMIT 1
                    """)
                    latest_entry = cursor.fetchone()
                    #logs.append(f"Latest entry: {latest_entry[3:7]}, Validation Status: {latest_entry[7]}, Datetime: {latest_entry[-1]}")
                    logs.append(f"Detected Entry: {first_name} {last_name}, IDSU: {idsu}, Department: {department}, Role: {role}, Validation Status: {validation_status}, Datetime: {latest_entry[-1]}")
            else:
                # Handle unknown face
                results.append((None, None, None, (x, y, w, h)))
                current_time = time.time()
                unknown_folder = "unknown_entry"
                os.makedirs(unknown_folder, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snapshot_path = os.path.join(unknown_folder, f"unknown_{timestamp}.jpg")

                # Always log the detection of an unknown face
                logs.append("Detected: Unknown face")

                # Save snapshot and log if 5-second interval has passed
                if current_time - last_unknown_snapshot_time >= 5:
                    cv2.imwrite(snapshot_path, frame)
                    cursor.execute("""
                        INSERT INTO pedestrian_entry_history (STUDENT_ID, FACULTY_ID, idsu, first_name, last_name, department, role, validation_status)
                        VALUES (NULL, NULL, 'Unknown', 'Unknown', 'Unknown', 'Unknown', 'Unknown', 'Unknown')
                    """)
                    con.commit()
                    logs.append(f"Unknown face detected. Snapshot saved to {snapshot_path}")
                    last_unknown_snapshot_time = current_time  # Update the last snapshot time

        return results, logs
    except Exception as e:
        current_time = time.time()
        if "Face could not be detected" in str(e):
            if current_time - last_print_time >= 5:
                logs.append("No face detected")
                last_print_time = current_time
        else:
            logs.append(f"Error in recognition: {e}")
        return [], logs
