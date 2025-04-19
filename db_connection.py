import mysql.connector

def database_connection():
    # Connect to MySQL database
    try:
        # Initial connection without database
        con = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="eve",
            password="137982"
        )
        cursor = con.cursor()
        database_name = "face_recognition"

        # Check if database exists
        cursor.execute(f"SHOW DATABASES LIKE '{database_name}'")
        result = cursor.fetchone()

        # Create database if it doesn't exist
        if not result:
            cursor.execute(f"CREATE DATABASE {database_name}")
            print(f"Database '{database_name}' created successfully")

        cursor.close()
        con.close()

        # Connect to specific database
        con = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="eve",
            password="137982",
            database=database_name,
            autocommit=False 
        )
        if con.is_connected():
            print("Successfully connected to the database")
        return con
    except mysql.connector.Error as err:
        print(f"Error: {err}")  # Log error
        return None

def close_database_connection(cursor, con):
    # Close cursor and connection
    if cursor:
        cursor.close()
    if con and con.is_connected():
        con.close()

def create_tables(con):
    # Create database tables
    try:
        cursor = con.cursor()
        # Table for students
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_enrolled_students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                img_path VARCHAR(100),
                first_name VARCHAR(20),
                last_name VARCHAR(25),
                validation_status VARCHAR(25),
                idSU VARCHAR(20),
                department VARCHAR(25),
                embedding BLOB,
                role VARCHAR(25)
            )
        """)
        # Table for faculty
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_faculty (
                id INT AUTO_INCREMENT PRIMARY KEY,
                createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                img_path VARCHAR(100),
                first_name VARCHAR(20),
                last_name VARCHAR(25),
                validation_status VARCHAR(25),
                idSU VARCHAR(20),
                department VARCHAR(25),
                embedding BLOB,
                role VARCHAR(25)
            )
        """)
        # Table for embedding dimensions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding (
                STUDENT_ID INT(20) NULL,
                FACULTY_ID INT(20) NULL,
                FACE_ID INT(20),
                DIM_NUM INT(20),
                VALUE FLOAT(35,30),
                FOREIGN KEY (STUDENT_ID) REFERENCES current_enrolled_students(id) ON DELETE CASCADE,
                FOREIGN KEY (FACULTY_ID) REFERENCES active_faculty(id) ON DELETE CASCADE
            )
        """)
        # Table for entry history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pedestrian_entry_history (
                STUDENT_ID INT(20) NULL,
                FACULTY_ID INT(20) NULL,
                ENTRY_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                IDSU VARCHAR(20),
                FIRST_NAME VARCHAR(20),
                LAST_NAME VARCHAR(25),
                DEPARTMENT VARCHAR(25),
                ROLE VARCHAR(25),
                VALIDATION_STATUS VARCHAR(25),
                FOREIGN KEY (STUDENT_ID) REFERENCES current_enrolled_students(id) ON DELETE CASCADE,
                FOREIGN KEY (FACULTY_ID) REFERENCES active_faculty(id) ON DELETE CASCADE
            )
        """)
        # Table for admin
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
        # Insert default admin credentials if not exists
        cursor.execute("""
            INSERT IGNORE INTO admin (username, password)
            VALUES ('admin', 'admin123')
        """)
        con.commit()  # Save changes
        cursor.close()
    except mysql.connector.Error as err:
        print(f"Error: {err}")  # Log error