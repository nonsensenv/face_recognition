import cv2
import os

# Common path within OpenCV's data directory
cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
if os.path.exists(cascade_path):
    print(f"File found at: {cascade_path}")
else:
    print("File not found in OpenCV data directory.")

# Check project directory
project_path = "~/Downloads/face_recognition_beta/haarcascade_frontalface_default.xml"
if os.path.exists(os.path.expanduser(project_path)):
    print(f"File found at: {project_path}")
else:
    print("File not found in project directory.")