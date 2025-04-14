from deepface import DeepFace
import matplotlib.pyplot as plt
import time

img1_path = "me/tristan5.jpg"
img2_path = "me/tristan7.jpg"

detectors = ['opencv', 'ssd']

for detector in detectors:

    obj = DeepFace.verify(img1_path, img2_path, detector_backend=detector, model_name="ArcFace", distance_metric="euclidean")

    start_time = time.time()
    img1_faces = DeepFace.extract_faces(img_path=img1_path, detector_backend=detector)
    img2_faces = DeepFace.extract_faces(img_path=img2_path, detector_backend=detector)

    if img1_faces:
        img1 = img1_faces[0]['face']
        plt.imshow(img1[:, :, ::-1])
        plt.show()

    if img2_faces:
        img2 = img2_faces[0]['face']
        plt.imshow(img2[:, :, ::-1])
        plt.show()
    print(obj)
    print(f"Detector: {detector}", time.time() - start_time, " seconds")
    print("--------------------------------------------------")