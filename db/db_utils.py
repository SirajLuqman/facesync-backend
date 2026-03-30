import mysql.connector
import pickle
import hashlib
from db.db_config import DB_CONFIG

# ---------------- Database Connection ----------------
def get_db_connection():
    """Create and return a new MySQL connection"""
    return mysql.connector.connect(**DB_CONFIG)


# ---------------- Admin Helpers ----------------

def insert_admin(username, email, admin_id_code, password):
    """
    Creates a new admin.
    HASHES the password before saving to ensure login works later.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 🚀 1. HASH THE PASSWORD BEFORE INSERTING
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        query = """
            INSERT INTO admins (username, email, admin_id_code, password_hash) 
            VALUES (%s, %s, %s, %s)
        """
        # 🚀 2. Use hashed_password instead of the raw password variable
        cursor.execute(query, (username, email, admin_id_code, hashed_password))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print(f"DEBUG: SQL Error - {err}")
        return False
    finally:
        cursor.close()
        conn.close()

def verify_admin_login(email_or_id, password):
    """
    Checks if the credentials match an admin in the database.
    """
    # 🚀 MUST match the hashing used in insert_admin and update_password
    hashed_input = hashlib.sha256(password.encode()).hexdigest()

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT * FROM admins WHERE (email=%s OR admin_id_code=%s) AND password_hash=%s"
    cursor.execute(query, (email_or_id, email_or_id, hashed_input))
    
    admin = cursor.fetchone()
    
    cursor.close()
    conn.close()
    return admin


def insert_person_if_not_exists(name, user_id_code, role=None):
    """
    Checks if a User ID already exists. 
    If yes, returns that ID. If no, creates a new record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Check if the User ID (e.g., S001) already exists
        query_check = "SELECT person_id FROM persons WHERE user_id_code=%s"
        cursor.execute(query_check, (user_id_code,))
        result = cursor.fetchone()
        
        if result:
            print(f"DEBUG: User {user_id_code} already exists.")
            person_id = result[0]
        else:
            # 2. Insert new person (Age is removed as per your request)
            query_insert = "INSERT INTO persons (name, user_id_code, role) VALUES (%s, %s, %s)"
            cursor.execute(query_insert, (name, user_id_code, role))
            conn.commit()
            person_id = cursor.lastrowid
            print(f"DEBUG: New user {name} inserted with ID {person_id}")
            
        return person_id

    except mysql.connector.Error as err:
        print(f"DATABASE ERROR: {err}")
        return None
    finally:
        cursor.close()
        conn.close()


def update_person_details(person_id, name, user_id_code=None, role=None):
    """
    Update the details of an existing person.
    REMOVED 'age' to match new table structure.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 1. We now update user_id_code instead of age
        query = "UPDATE persons SET name=%s, user_id_code=%s, role=%s WHERE person_id=%s"
        cursor.execute(query, (name, user_id_code, role, person_id))
        conn.commit()
        print(f"DEBUG: Person {person_id} updated successfully.")
    except mysql.connector.Error as err:
        print(f"UPDATE ERROR: {err}")
    finally:
        cursor.close()
        conn.close()


def delete_person_by_id(person_id):
    """
    Delete a person and all their embeddings from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete all embeddings for this person first
    cursor.execute("DELETE FROM face_embeddings WHERE person_id=%s", (person_id,))
    
    # Delete the person record
    cursor.execute("DELETE FROM persons WHERE person_id=%s", (person_id,))
    
    conn.commit()
    cursor.close()
    conn.close()


# ---------------- Face Embedding Helpers ----------------
def insert_embedding(person_id, embedding, source_image=None):
    """
    Insert a face embedding for a person.
    embedding: numpy array
    source_image: optional filename
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Serialize numpy array using pickle
    embedding_bytes = pickle.dumps(embedding)
    
    cursor.execute(
        "INSERT INTO face_embeddings (person_id, embedding, source_image) VALUES (%s, %s, %s)",
        (person_id, embedding_bytes, source_image)
    )
    
    conn.commit()
    cursor.close()
    conn.close()


def fetch_all_embeddings():
    """
    Fetch all embeddings from the database and return as a list of tuples (person_id, numpy_array_embedding)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT person_id, embedding FROM face_embeddings")
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    embeddings = []
    for person_id, emb_bytes in results:
        emb = pickle.loads(emb_bytes)
        embeddings.append((person_id, emb))
    
    return embeddings

def fetch_embeddings_by_person(pid):
    """Return all embeddings (with ids) for a given person_id"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT embedding_id, embedding, source_image FROM face_embeddings WHERE person_id=%s", (pid,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows

def delete_embedding_by_id(eid):
    """Delete a specific embedding by embedding_id"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("DELETE FROM face_embeddings WHERE embedding_id=%s", (eid,))
    conn.commit()
    cur.close(); conn.close()
