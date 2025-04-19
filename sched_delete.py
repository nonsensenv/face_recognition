import os
import time
import mysql.connector
from datetime import datetime, timedelta

unknown_folder = "/home/eve/Downloads/face_recognition/face_recognition/unknown_entry"
days_limit = 30 #days until deletion

conn = mysql.connector.connect(
    host="localhost",
    port="3306",
    user="eve",
    password="137982",
    database="face_recognition"
)
cursor = conn.cursor()

cutoff_date = datetime.now() - timedelta(days=days_limit)

cursor.execute("SELECT COUNT(*) FROM pedestrian_entry_history WHERE ENTRY_TIME < %s", (cutoff_date,))
db_count = cursor.fetchone()[0]
if db_count == 0:
    print(f"No database entries older than {days_limit} days.")
else:
    cursor.execute("DELETE FROM pedestrian_entry_history WHERE ENTRY_TIME < %s", (cutoff_date,))
    conn.commit()
    print(f"Deleted {db_count} database entries older than {days_limit} days.")

if not os.path.exists(unknown_folder):
    print(f"Folder '{unknown_folder}' does not exist. Skipping file deletion.")
else:
    file_deleted = False
    for filename in os.listdir(unknown_folder):
        filepath = os.path.join(unknown_folder, filename)
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff_date:
                os.remove(filepath)
                print(f"Deleted {filepath}")
                file_deleted = True
    if not file_deleted:
        print(f"No files older than {days_limit} days in the folder '{unknown_folder}'.")

cursor.close()
conn.close()
