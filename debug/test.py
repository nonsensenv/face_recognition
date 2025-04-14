from deepface import DeepFace
import cv2
import os

import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
# Directory containing reference images for face recognition
reference_img_dir = "me"

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract faces from the frame
    faces = DeepFace.extract_faces(img_path=frame, detector_backend='ssd', enforce_detection=False)
    for face in faces:
        face_img = face['face']
        print(face_img)
        facial_area = face['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Find the most similar image from the reference directory
        df = DeepFace.find(img_path=face_img, db_path=reference_img_dir, model_name="ArcFace", distance_metric="euclidean", enforce_detection=False)
        if df:
            most_similar_img_path = df[0]['identity']
            
            # Verify the detected face with the most similar image
            result = DeepFace.verify(img1_path=most_similar_img_path, img2_path=face_img, model_name="ArcFace", distance_metric="euclidean", enforce_detection=False)
            # print(result)
            
            # Display the filename on top of the face frame
            filename = most_similar_img_path[0]
            cv2.putText(frame, filename, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame with rectangles around detected faces
    cv2.imshow("frame", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
