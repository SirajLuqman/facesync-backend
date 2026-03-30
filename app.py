from flask import Flask, request, jsonify
import cv2
import io
import csv
import numpy as np
from mtcnn import MTCNN
import os
import time
import hashlib
import random
from scipy.spatial.distance import cosine
from keras_facenet import FaceNet
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from flask import send_file


# Import updated helpers
from db.db_utils import (
    insert_admin,
    verify_admin_login,
    insert_person_if_not_exists, 
    insert_embedding, 
    get_db_connection,
    fetch_all_embeddings
)

# ------------------------------
# INITIALIZE MODELS & CONFIG
# ------------------------------
embedder = FaceNet()
detector = MTCNN()
app = Flask(__name__)
RECOGNITION_THRESHOLD = 0.28   
GALLERY_RECOGNITION_THRESHOLD = 0.28
KNOWN_EMBEDDINGS = []

# 1. Configure Email (Use an App Password, not your regular Gmail password)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'shopanytime3@gmail.com'
app.config['MAIL_PASSWORD'] = 'elam idcc dvvw rimh' # 16 character code
mail = Mail(app)

otp_storage = {}

# Ensure the folder for captured faces exists
UPLOAD_FOLDER = 'captured_faces'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------------------
# HELPER
# ------------------------------
def refresh_embeddings_cache():
    global KNOWN_EMBEDDINGS
    print("🔄 Refreshing Embeddings Cache from DB...")
    KNOWN_EMBEDDINGS = fetch_all_embeddings()
    print(f"✅ Cache updated: {len(KNOWN_EMBEDDINGS)} signatures loaded.")


def extract_face(img, box):
    try:
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        face = img[y:y+h, x:x+w]

        if face.size == 0:
            print("extract_face: Empty crop!")
            return None

        face = cv2.resize(face, (160, 160))
        face = face.astype("float32")

        return np.expand_dims(face, axis=0)

    except Exception as e:
        print(f"Extraction Error: {e}")
        return None

def get_embedding(face_pixels):
    if face_pixels is None:
        return None

    # Verify pixel data is unique
    print(f"DEBUG INPUT: mean={np.mean(face_pixels):.6f}, std={np.std(face_pixels):.6f}")

    embedding = embedder.embeddings(face_pixels)[0]
    print(f"DEBUG RAW: std={np.std(embedding):.6f}, sum={np.sum(embedding):.4f}")

    if np.all(embedding == 0) or np.std(embedding) < 1e-5:
        print("Invalid embedding — rejecting.")
        return None

    normalized = embedding / (np.linalg.norm(embedding) + 1e-6)
    return normalized

def match_identity(embedding, threshold=RECOGNITION_THRESHOLD):
    global KNOWN_EMBEDDINGS
    if not KNOWN_EMBEDDINGS:
        refresh_embeddings_cache()
    
    db_raw_data = KNOWN_EMBEDDINGS

    # 1. Group embeddings by person_id
    temp_groups = {}
    for p_id, db_emb in db_raw_data:
        if p_id not in temp_groups:
            temp_groups[p_id] = []
        temp_groups[p_id].append(db_emb)

    best_dist = 1.0
    identity_id = None

    # 2. Compare live face against the MEAN (Average) of each person
    for p_id, embs in temp_groups.items():
        # Calculate the "Master Signature" for this person
        mean_emb = np.mean(embs, axis=0)
        # Re-normalize the mean for mathematical consistency
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
        
        dist = cosine(embedding, mean_emb)
        
        if dist < best_dist:
            best_dist = dist
            identity_id = p_id

    # 3. Final Threshold Check
    if identity_id and best_dist < threshold:
        print(f"DEBUG: Match confirmed for ID {identity_id} with Dist {best_dist:.4f}")
        return f"User ID: {identity_id}"
    
    print(f"DEBUG: No match. Closest was ID {identity_id} at Dist {best_dist:.4f}")
    return "Unknown"

def save_log(person_id, status, is_success, image_name=None):
    """Inserts a scan record into the access_logs table, now with image support."""
    try:
        db = get_db_connection()
        cursor = db.cursor()
        
        # Updated query to include the image_path column
        query = "INSERT INTO access_logs (person_id, status, is_success, image_path) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (person_id, status, is_success, image_name))
        
        db.commit()
        print(f"📡 Log Saved: {status} | Image: {image_name}")
        cleanup_old_logs()
    except Exception as e:
        print(f"❌ Failed to save log: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        db.close()

def cleanup_old_logs():
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        
        # 1. Find the filenames of logs that are about to be deleted
        find_old_images = """
            SELECT image_path FROM access_logs 
            WHERE log_id NOT IN (
                SELECT log_id FROM (
                    SELECT log_id FROM access_logs ORDER BY timestamp DESC LIMIT 1000
                ) as tmp
            ) AND image_path IS NOT NULL
        """
        cursor.execute(find_old_images)
        old_images = cursor.fetchall()

        # 2. Delete the physical files
        for row in old_images:
            file_path = os.path.join('static/unknown_faces', row['image_path'])
            if os.path.exists(file_path):
                os.remove(file_path)

        # 3. Now run your existing DELETE query
        cleanup_query = "DELETE FROM access_logs WHERE log_id NOT IN (SELECT log_id FROM (SELECT log_id FROM access_logs ORDER BY timestamp DESC LIMIT 1000) as tmp)"
        cursor.execute(cleanup_query)
        conn.commit()
    except Exception as e:
        print(f"Cleanup Error: {e}")
    finally:
        conn.close()

# ------------------------------
# API ENDPOINTS
# ------------------------------

# User Registration
@app.route('/register_user', methods=['POST'])
def register_user():
    try:
        # 1. Extract Text Data
        name = request.form.get('name')
        user_id_code = request.form.get('user_id_code')
        role = request.form.get('role')
        files = request.files.getlist('images')

        if not name or not user_id_code or not files:
            return jsonify({"status": "error", "message": "Missing required data"}), 400

        # --- DATABASE ID PRE-CHECK ---
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT name FROM persons WHERE user_id_code=%s", (user_id_code,))
        existing_id = cursor.fetchone()
        
        if existing_id:
            conn.close()
            return jsonify({
                "status": "error", 
                "message": f"ID {user_id_code} is already assigned to {existing_id['name']}."
            }), 409

        # --- BIOMETRIC DUPLICATE CHECK ---
        # Read the first image into memory to check if this face already exists
        first_file = files[0]
        file_bytes = np.frombuffer(first_file.read(), np.uint8)
        first_file.seek(0) # Reset pointer for saving later
        
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces:
            face_pixels = extract_face(rgb, faces[0]["box"])
            new_emb = get_embedding(face_pixels)

            if new_emb is not None:
                # Use Cache for Duplicate Check (High Speed)
                global KNOWN_EMBEDDINGS
                if not KNOWN_EMBEDDINGS: 
                    refresh_embeddings_cache()
                
                for p_id, db_emb in KNOWN_EMBEDDINGS:
                    dist = cosine(new_emb, db_emb)
                    if dist < RECOGNITION_THRESHOLD:
                        cursor.execute("SELECT name FROM persons WHERE person_id=%s", (p_id,))
                        real_owner = cursor.fetchone()
                        conn.close()
                        owner_name = real_owner['name'] if real_owner else "Unknown User"
                        return jsonify({
                            "status": "error", 
                            "message": f"This face is already enrolled as {real_owner['name']}."
                        }), 409
        
        conn.close() 

        # --- PROCEED WITH ACTUAL REGISTRATION ---
        # Step A: Create the person record in the 'persons' table
        person_id = insert_person_if_not_exists(name, user_id_code, role)
        embedding_count = 0
        
        # Step B: Process each uploaded image
        for i, file in enumerate(files):
            if file:
                filename = f"{user_id_code}_{i}.jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                # 1. Save temporarily
                file.save(filepath)

                # 2. Read from disk for MTCNN/FaceNet
                img = cv2.imread(filepath)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = detector.detect_faces(img_rgb)
                    
                    if faces:
                        face_pixels = extract_face(img_rgb, faces[0]["box"])
                        embedding = get_embedding(face_pixels)
                        
                        # 3. Store the math (embedding) in the DB
                        insert_embedding(person_id, embedding, filename)
                        embedding_count += 1

                # 4. CLEANUP: Delete file only AFTER we are done reading it
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"🗑️ Deleted temp registration file: {filename}")

        # --- UPDATE SYSTEM CACHE ---
        # Reload the global KNOWN_EMBEDDINGS so the new user is recognized instantly
        refresh_embeddings_cache()

        return jsonify({
            "status": "success",
            "message": f"User {name} enrolled with {embedding_count} face signatures",
            "person_id": person_id
        }), 200

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/signup", methods=["POST"])
def api_signup():
    data = request.json
    success = insert_admin(
        data.get("username"), 
        data.get("email"), 
        data.get("admin_id_code"), 
        data.get("password")
    )
    if success:
        return jsonify({"status": "success", "message": "Admin created successfully"}), 201
    return jsonify({"status": "error", "message": "Registration failed"}), 400

@app.route("/recognize", methods=["POST"])
def api_recognize():
    file = request.files.get("image")
    if not file: 
        return jsonify({"error": "No image"}), 400

    # 1. Process the incoming image from Android
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = detector.detect_faces(rgb)
    if not faces:
        print("🔍 UI Update: Sending 'No Face' status (422)")
        return jsonify({"status": "no_face", "message": "No face detected"}), 422 

    # 2. Generate embedding for the live face
    face_pixels = extract_face(rgb, faces[0]["box"])
    live_embedding = get_embedding(face_pixels)
    
    # 3. Search the Database using the Consensus Matcher
    # match_identity returns "User ID: X" or "Unknown"
    result = match_identity(live_embedding, threshold=RECOGNITION_THRESHOLD)

    # 4. Process the result logic (Fixing the NameError scope)
    if result != "Unknown":
        # Extract the ID safely
        matched_person_id = int(result.split(": ")[1])

        # RE-CALCULATE DISTANCE: This defines 'min_dist' properly for this scope
        global KNOWN_EMBEDDINGS
        person_embs = [emb for p_id, emb in KNOWN_EMBEDDINGS if p_id == matched_person_id]
        mean_emb = np.mean(person_embs, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
        min_dist = cosine(live_embedding, mean_emb)

        # 5. Get real details from 'persons' table
        # We check min_dist here now that it is defined
        if min_dist < RECOGNITION_THRESHOLD:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT name, user_id_code, role FROM persons WHERE person_id=%s", (matched_person_id,))
            user = cursor.fetchone()
            conn.close()
            
            if user:
                save_log(matched_person_id, "Authorized", True, None)
                confidence_score = round((1 - min_dist) * 100, 2)
                
                print(f"DEBUG: Match Found! {user['name']} ({confidence_score}%)")
                return jsonify({
                    "status": "success",
                    "name": user['name'],
                    "user_id": user['user_id_code'],
                    "role": user['role'],
                    "confidence": str(confidence_score)
                })

    # 6. Fallback: If no match, distance too high, or result was "Unknown"
    # Note: We use a generic message here if min_dist wasn't even calculated
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    img_name = f"intruder_{timestamp}.jpg"
    
    # Ensure static directory exists
    target_dir = 'static/unknown_faces'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    cv2.imwrite(os.path.join(target_dir, img_name), img) 
    save_log(None, "Access Denied", False, img_name)
    
    print(f"DEBUG: No Match Found (User not recognized)")
    return jsonify({"status": "unknown", "message": "User not recognized"}), 404

@app.route("/admin_login", methods=["POST"])
def api_admin_login():
    data = request.json
    email_or_id = data.get("email_or_id")
    password = data.get("password")

    MASTER_ID = "Admin" 
    MASTER_PASS = "Admin123@"

    if email_or_id == MASTER_ID and password == MASTER_PASS:
        return jsonify({
            "status": "success", 
            "message": "Master Access Granted",
            "role": "super_admin"
        }), 200

    admin = verify_admin_login(email_or_id, password)

    if admin:
        return jsonify({"status": "success", "message": "Welcome back!"}), 200
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401


@app.route('/recognize_multiple', methods=['POST'])
def recognize_multiple():
    if 'images' not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist('images')
    valid_embeddings = []

    print("\n--- GALLERY ANALYSIS: CONSENSUS MODE ---")
    
    # 1. Collect and filter all face signatures from the gallery upload
    for i, file in enumerate(files):
        file_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces and faces[0]['confidence'] > 0.95:
            face_pixels = extract_face(rgb, faces[0]["box"])
            emb = get_embedding(face_pixels)
            if emb is not None:
                valid_embeddings.append(emb)

    if not valid_embeddings:
        print("🔍 Gallery Result: No faces detected (422)")
        return jsonify({"status": "no_face", "message": "No clear faces found in gallery"}), 422

    # 2. CREATE A MASTER PROBE: Average the gallery images together
    gallery_mean_emb = np.mean(valid_embeddings, axis=0)
    gallery_mean_emb = gallery_mean_emb / (np.linalg.norm(gallery_mean_emb) + 1e-6)

    # 3. COMPARE MASTER PROBE AGAINST DB MEAN PROFILES
    # match_identity now handles the averaging of the database side
    result = match_identity(gallery_mean_emb, threshold=GALLERY_RECOGNITION_THRESHOLD)

    if result != "Unknown":
        matched_person_id = int(result.split(": ")[1])
        
        # Calculate accurate confidence
        global KNOWN_EMBEDDINGS
        db_raw_data = KNOWN_EMBEDDINGS
        person_embs = [emb for p_id, emb in db_raw_data if p_id == matched_person_id]
        db_mean_emb = np.mean(person_embs, axis=0)
        db_mean_emb = db_mean_emb / (np.linalg.norm(db_mean_emb) + 1e-6)
        
        final_dist = cosine(gallery_mean_emb, db_mean_emb)
        confidence_score = round((1 - final_dist) * 100, 2)

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT name, user_id_code, role FROM persons WHERE person_id=%s", (matched_person_id,))
        user = cursor.fetchone()
        conn.close()

        if user:
            print(f"✅ GALLERY MATCH: {user['name']} ({confidence_score}%)")
            return jsonify({
                "status": "success",
                "name": user['name'],
                "user_id": user['user_id_code'],
                "role": user['role'],
                "confidence": str(confidence_score)
            })

    print("❌ GALLERY RESULT: Unknown Person")
    return jsonify({"status": "unknown", "message": "No match found"}), 404
  
@app.route('/recognize_live', methods=['POST'])
def recognize_live():
    if 'image' not in request.files:
        return jsonify({"name": "Error", "confidence": 0.0}), 400

    file = request.files['image']
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"name": "Invalid Image", "confidence": 0.0}), 400

    # 1. Detection at 480p for better landmark accuracy
    detect_w, detect_h = 480, 360
    small_frame = cv2.resize(frame, (detect_w, detect_h))
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_small)

    if not faces:
        print("🔍 UI Update: Sending 'No Face' status (422)")
        return jsonify({"status": "no_face", "message": "No face detected"}), 422 # Changed from 404

    # 2. Map coordinates & Extract
    x, y, w, h = faces[0]["box"]
    scale_x, scale_y = frame.shape[1]/detect_w, frame.shape[0]/detect_h
    box = [int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_img = extract_face(rgb_frame, box) 

    # 3. Generate & Normalize Embedding
    embedding = embedder.embeddings(face_img)[0]
    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

    # 4. CONSENSUS MATCHING
    result = match_identity(embedding, threshold=RECOGNITION_THRESHOLD)

    # 5. Fixed Response Logic
    if result != "Unknown":
        matched_person_id = int(result.split(": ")[1])
        
        # We need to find the distance again for the confidence score
        global KNOWN_EMBEDDINGS
        person_embs = [emb for p_id, emb in KNOWN_EMBEDDINGS if p_id == matched_person_id]
        mean_emb = np.mean(person_embs, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
        
        # Calculate distance properly here to fix the NameError
        current_dist = cosine(embedding, mean_emb)
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT name FROM persons WHERE person_id=%s", (matched_person_id,))
        user_record = cursor.fetchone()
        conn.close()
        
        resolved_name = user_record['name'] if user_record else "Authorized"
        
        # Use current_dist for confidence calculation
        confidence = max(0, (1.0 - (current_dist / 0.6)) * 100) 
        
        print(f"🎯 Live Match Found: {resolved_name} (Dist: {current_dist:.4f})")
        return jsonify({
            "name": str(resolved_name),
            "confidence": round(float(confidence), 2)
        })

    print("🔍 Result: Unknown / No Match")
    return jsonify({"name": "Unknown", "confidence": 0.0})
    
@app.route('/request-otp', methods=['POST'])
def request_otp():
    data = request.json
    admin_id_input = data.get('admin_id') # Matches Android 'map.put("admin_id", id)'
    email_input = data.get('email')

    # 🛑 CRITICAL: Check the database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM admins WHERE admin_id_code=%s AND email=%s", 
                   (admin_id_input, email_input))
    admin = cursor.fetchone()
    conn.close()

    if not admin:
        return jsonify({"status": "error", "message": "Admin credentials not found"}), 404
    
    # Proceed with OTP generation if admin is found
    otp = str(random.randint(100000, 999999))
    otp_storage[email_input] = {
        "otp": otp,
        "expiry": time.time() + 300  # Expires in 5 minutes
    }

    try:
        msg = Message("FaceSync Password Reset", 
                      sender=app.config['MAIL_USERNAME'], 
                      recipients=[email_input])
        msg.body = f"Your verification code is: {otp}\nThis code expires in 5 minutes."
        mail.send(msg)
        return jsonify({"status": "success", "message": "OTP sent"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json
    email = data.get('email')
    user_otp = data.get('otp')

    # Get the stored OTP data for this email
    stored_data = otp_storage.get(email)

    if not stored_data:
        return jsonify({"status": "error", "message": "No OTP request found"}), 400

    # Check if OTP matches and hasn't expired (5 mins)
    if stored_data['otp'] == user_otp:
        if time.time() < stored_data['expiry']:
            return jsonify({"status": "success", "message": "OTP Verified"}), 200
        else:
            return jsonify({"status": "error", "message": "OTP Expired"}), 400
    
    return jsonify({"status": "error", "message": "Invalid OTP code"}), 401

@app.route('/update-password', methods=['POST'])
def update_password():
    data = request.json
    email = data.get('email')
    new_password = data.get('new_password')

    # Hash the password to match your admin_login logic
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # SQL Update command
        cursor.execute("UPDATE admins SET password_hash=%s WHERE email=%s", (hashed_password, email))
        conn.commit()
        conn.close()
        
        print(f"🔐 Password successfully updated for {email}")
        return jsonify({"status": "success", "message": "Password changed"}), 200
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_users/<string:type>', methods=['GET'])
def get_users(type):
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        if type == "user":
            # CAST converts the INT person_id into a String for Android
            cursor.execute("SELECT name, role, CAST(user_id_code AS CHAR) as id FROM persons")
        else:
            # CAST converts the INT admin_id into a String for Android
            cursor.execute("SELECT username as name, 'Admin' as role, CAST(admin_id_code AS CHAR) as id FROM admins")
            
        results = cursor.fetchall()
        cursor.close()
        return jsonify(results)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify([]), 500
    finally:
        conn.close()

@app.route('/delete_user/<string:user_type>/<string:user_id>', methods=['DELETE'])
def delete_user(user_type, user_id):
    db = get_db_connection()
    cursor = db.cursor()
    
    try:
        if user_type == "user":
            # Match against the office code column
            cursor.execute("DELETE FROM face_embeddings WHERE person_id = (SELECT person_id FROM persons WHERE user_id_code = %s)", (user_id,))
            cursor.execute("DELETE FROM persons WHERE user_id_code = %s", (user_id,))
        else:
            # Match against the office code column
            cursor.execute("DELETE FROM admins WHERE admin_id_code = %s", (user_id,))
            
        db.commit()
        return jsonify({"status": "success", "message": "Deleted successfully"}), 200
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cursor.close()
        db.close()

@app.route('/get_logs', methods=['GET'])
def get_logs():
    """Fetches scan history including User ID Code and Image Path for the Android UI."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        # 1. Added p.user_id_code and l.image_path to the SELECT
        # 2. Using COALESCE for user_id_code to show 'N/A' for unknown users
        query = """
            SELECT 
                l.log_id, 
                COALESCE(p.name, 'Unknown') as name, 
                COALESCE(p.user_id_code, 'N/A') as user_id_code,
                l.status, 
                l.is_success, 
                l.image_path,
                l.timestamp 
            FROM access_logs l
            LEFT JOIN persons p ON l.person_id = p.person_id
            ORDER BY l.timestamp DESC 
            LIMIT 100
        """
        cursor.execute(query)
        logs = cursor.fetchall()
        
        # Convert datetime objects to strings
        for log in logs:
            if log['timestamp']:
                log['timestamp'] = log['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            
        return jsonify(logs), 200
    except Exception as e:
        print(f"❌ Error fetching logs: {e}")
        return jsonify([]), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/', methods=['GET'])
def check_health():
    """Simple health check for Android to verify server is reachable."""
    return jsonify({"status": "online", "message": "FaceSync Server Active"}), 200

@app.route('/clear_logs', methods=['DELETE', 'POST'])
def clear_logs():
    """Wipes all records from the access_logs table."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # 1. Delete physical intruder images first to save disk space
        img_folder = 'static/unknown_faces'
        for filename in os.listdir(img_folder):
            file_path = os.path.join(img_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        # 2. Clear the MySQL table
        cursor.execute("TRUNCATE TABLE access_logs") 
        conn.commit()
        
        print("🧹 Database Maintenance: Access logs cleared by Admin.")
        return jsonify({"status": "success", "message": "Log history wiped"}), 200
    except Exception as e:
        print(f"❌ Clear Logs Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/export_logs', methods=['GET'])
def export_logs():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT l.timestamp, COALESCE(p.name, 'Unknown') as name, 
                   l.status, l.is_success 
            FROM access_logs l 
            LEFT JOIN persons p ON l.person_id = p.person_id 
            ORDER BY l.timestamp DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Timestamp', 'User Name', 'Status', 'Access Granted']) 
        
        for row in rows:
            writer.writerow([row['timestamp'], row['name'], row['status'], row['is_success']])

        output.seek(0)
        
        # Add a date to the filename
        current_date = time.strftime("%Y%m%d-%H%M")
        file_name = f"FaceSync_Report_{current_date}.csv"

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=file_name
        )
    except Exception as e:
        print(f"Export Error: {e}")
        return "Failed to generate report", 500
    finally:
        conn.close()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
