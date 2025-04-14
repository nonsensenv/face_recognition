import threading
import subprocess

def run_beta():
    subprocess.run(["python3", "/home/eve/Downloads/face_recognition/face_recognition/beta.py"])

def run_gui():
    subprocess.run(["python3", "/home/eve/Downloads/face_recognition/face_recognition/gui.py"])

if __name__ == "__main__":
    beta_thread = threading.Thread(target=run_beta)
    gui_thread = threading.Thread(target=run_gui)

    beta_thread.start()
    gui_thread.start()

    beta_thread.join()
    gui_thread.join()
