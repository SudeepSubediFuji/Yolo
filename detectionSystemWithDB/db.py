import os
import sqlite3

import face_recognition
import numpy as np
from check_photo import photo_conversion
from app_utils import parse_clean_name
DB_FILE = "faces.db"
KNOWN_FACES_DIR = "./known_faces"


def get_db_connection():
    ## check_same_thread → この引数をFalseに設定するとスレッドのチェック処理を行わず接続する
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    # 以下のコマンドを実行したら下で列情報を取得しやすい
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # users table -> id(pk),name,created_at

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)
    """)

    # face_embeddings -> users_id(fk),face_embedding(BLOB),source_path,id(pk)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    users_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    source_path TEXT NOT NULL,
    FOREIGN KEY(users_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()

def scan_and_sync_folder(folder_path=KNOWN_FACES_DIR):
    init_db()
    photo_conversion()
    conn = get_db_connection()
    cursor = conn.cursor()
    report = {"loaded": 0, "failed": [], "skipped": 0}

    ## フォルダパスをチェックする
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return report

    image_extension = (".jpeg", ".jpg", ".png", ".webp", ".heic", ".heif")
    files_path = [
        f for f in os.listdir(folder_path) if f.lower().endswith(image_extension)
    ]

    for file in files_path:
        image_path = os.path.join(folder_path, file)
        clean_name = parse_clean_name(image_path)

        cursor.execute("SELECT id FROM face_embeddings WHERE source_path = ?", (file,))

        if cursor.fetchone():
            report["skipped"] += 1
            continue

        try:
            image = face_recognition.load_image_file(image_path)
            face_location = face_recognition.face_locations(image)

            if not face_location:
                report["failed"].append(
                    {
                        "file": file,
                        "reason": "No face detected(check lighting,or head angle)",
                    }
                )
                continue

            encoding = face_recognition.face_encodings(image, face_location)[0]
            vector_blob = encoding.astype(np.float64).tobytes()

            # insert or get get parent user id
            cursor.execute("INSERT OR IGNORE INTO users(name) VALUES(?)", (clean_name,))
            cursor.execute("SELECT id FROM users WHERE name = ?", (clean_name,))
            user_id = cursor.fetchone()["id"]

            # store valid vectors in SQLite
            cursor.execute(
                "INSERT INTO face_embeddings(users_id, source_path, embedding) VALUES(?,?,?)",
                (user_id, file, vector_blob),
            )
            conn.commit()
            report["loaded"] += 1

        except Exception as e:
            report["failed"].append({"file": file, "reason": str(e)})

    conn.close()
    return report


def load_verified_profiles_from_db():
    """
    Reads verified vectors from SQLite into memory structures for inference.
    Returns:
        face_database: { embedding_id: np.ndarray(128,) }
        name_database: { embedding_id: "sudhsrth" }
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.id AS embed_id, u.name, e.embedding
        FROM face_embeddings e
        JOIN users u ON e.users_id = u.id
    """)
    rows = cursor.fetchall()
    conn.close()

    face_database = {}
    name_database = {}

    for row in rows:
        embed_id = row["embed_id"]
        vector = np.frombuffer(row["embedding"], dtype=np.float64)
        face_database[embed_id] = vector
        name_database[embed_id] = row["name"]

    return face_database, name_database


def unknown_profile_insert():
    # here i think there needs to be a process that not only update the cache or save the unknown to the db using the update method for Suspicious person
    ##　new db with similar column name as the users(reusing the face_embedding db if possible)
    return
