from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import PyPDF2
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import os
import time
import threading
from ultralytics import YOLO
import dlib
import whisper
import sounddevice as sd
import numpy as np
import queue
import librosa
import wave
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import sounddevice as sd
import whisper
import google.generativeai as genai
from flask_mail import Mail, Message
import smtplib
import random
import ssl

import sys
import gdown


load_dotenv()



app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

genai.configure(api_key=os.getenv('GENAI_API_KEY'))
model2 = genai.GenerativeModel("gemini-1.5-flash")
yolo_file_id = "1wDvhyrYQdqTsLbnWneW5O3WreXfGd2j2"

yolo_model_path = "models/yolo11n.pt"

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Check if YOLO model exists, otherwise download
if not os.path.exists(yolo_model_path):
    print("Downloading yolo11n.pt from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={yolo_file_id}", yolo_model_path, quiet=False, fuzzy=True)
    print("Download complete!")
else:
    print("YOLO model already exists. Skipping download.")

# Load YOLO model
model = YOLO(yolo_model_path)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS'))
firebase_admin.initialize_app(cred)
db = firestore.client()


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html', tab='login')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html', tab='aboutus')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('index.html', tab='signup')

        user_ref = db.collection('users').document(username)
        if user_ref.get().exists:
            flash('Username already exists!', 'danger')
            return render_template('index.html', tab='signup')

        user_ref.set({
            'username': username,
            'email': email,
            'password': generate_password_hash(password)
        })
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('index'))
    return render_template('index.html', tab='signup')


app.secret_key1 = os.urandom(24)

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") 
def send_otp(email):
    otp = str(random.randint(100000, 999999))
    session['otp'] = otp  # Store OTP in session
    session['email'] = email  # Store email in session

    subject = "Your OTP Code"
    body = f"Your OTP for verification is {otp}. Please do not share this with anyone."
    message = f"Subject: {subject}\n\n{body}"

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, email, message)
        server.quit()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

@app.route("/send-otp", methods=["POST"])
def send_otp_route():
    data = request.json
    email = data.get("email")

    if not email:
        return jsonify({"error": "Email is required"}), 400

    if send_otp(email):
        return jsonify({"message": "OTP sent successfully"}), 200
    else:
        return jsonify({"error": "Failed to send OTP"}), 500

@app.route("/verify-otp", methods=["POST"])
def verify_otp_route():
    data = request.json
    user_otp = data.get("otp")

    if not user_otp:
        return jsonify({"error": "OTP is required"}), 400

    if user_otp == session.get("otp"):
        return jsonify({"message": "OTP verified successfully"}), 200
    else:
        return jsonify({"error": "Invalid OTP"}), 400
    
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user_ref = db.collection('users').document(username)
        user = user_ref.get()
        if user.exists:
            user_data = user.to_dict()
            if check_password_hash(user_data['password'], password):
                session['username'] = username
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Incorrect password.', 'danger')
        else:
            flash('User does not exist.', 'danger')
    return render_template('index.html', tab='login')


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please log in first!', 'danger')
        return redirect(url_for('index'))

    username = session['username']
    tests = [doc.to_dict() for doc in db.collection('tests').where('username', '==', username).stream()]
    return render_template('dashboard.html', username=username, tests=tests)


def generate_questions(text, num_questions, marks, level):
    """Generates questions using generative AI."""
    genai.configure(api_key=os.getenv('GENAI_API_KEY'))
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Generate {num_questions} {level} questions based on the following text: {text}. "
        f"Each question should be worth {marks} marks. No answers or hints."
        f"just give questions no need of the other lines like : Here are 10 easy, multiple-choice questions worth 10 marks each, based on the provided text:"
        f"Dont give question numbers"
    )
    response = model.generate_content(prompt)
    return [q.strip() for q in response.text.split('\n') if q.strip()]


@app.route('/questions_generation', methods=['GET', 'POST'])
def questions_generation():
    if 'username' not in session:
        flash('Please log in first!', 'danger')
        return redirect(url_for('index'))

    username = session['username']
    tests = [doc.to_dict() for doc in db.collection('tests').where('username', '==', username).stream()]
    questions = []

    if request.method == 'POST':
        input_type = request.form.get('input_type')
        num_questions = int(request.form.get('num_questions'))
        marks = int(request.form.get('marks'))
        level = request.form.get('level')

        if input_type == 'pdf':
                uploaded_file = request.files.get('file')
                if uploaded_file:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
                    uploaded_file.save(file_path)
                    pdf_reader = PyPDF2.PdfReader(file_path)
                    text = "".join(page.extract_text() for page in pdf_reader.pages)
                    questions = generate_questions(text, num_questions, marks, level)

        elif input_type == 'text':
                input_text = request.form.get('input_text')
                if input_text:
                    questions = generate_questions(input_text, num_questions, marks, level)
    print("Generated Questions:", questions)

    return render_template('questions_generation.html', questions=questions, username=username, tests=tests)

from flask import request, flash, redirect, url_for
import json
from uuid import uuid4
from firebase_admin import firestore



import json  # Ensure json is imported

@app.route('/save_test', methods=['POST'])
def save_test():
    if 'username' not in session:
        flash('Please log in first!', 'danger')
        return redirect(url_for('index'))

    username = session['username']
    test_code = str(uuid4())[:8]  # Generate a unique test code

    try:
        # Fetch test details
        marks = request.form.get('marks')
        level = request.form.get('level')
        topic = request.form.get('topic')
        date = request.form.get('date')
        start = request.form.get('start')
        end = request.form.get('end')
        questions_json = request.form.get('questions')  # Get questions from form

        # Validate required fields
        if not all([marks, level, topic, date, questions_json]):
            flash("All fields and questions are required!", "danger")
            return redirect(url_for('questions_generation'))

        # Convert JSON string to Python list
        try:
            questions = json.loads(questions_json)  # Parse JSON string to list
        except json.JSONDecodeError as e:
            flash("Invalid JSON format for questions!", "danger")
            print("JSON Decode Error:", str(e))
            return redirect(url_for('questions_generation'))

        # Fetch user email from Firestore
        user_ref = db.collection('users').document(username).get()
        if not user_ref.exists:
            flash("User not found!", "danger")
            return redirect(url_for('questions_generation'))

        user_email = user_ref.to_dict().get('email')
        if not user_email:
            flash("User email not found!", "danger")
            return redirect(url_for('questions_generation'))

        # Prepare test data for Firestore
        test_data = {
            'username': username,
            'test_code': test_code,
            'topic': topic,
            'date': date,
            'start_time': start,
            'end_time': end,
            'marks': int(marks),
            'level': level,
            'questions': questions,  # Store questions as a JSON array
            'created_at': firestore.SERVER_TIMESTAMP,
        }

        # Save to Firestore
        db.collection('tests').document(test_code).set(test_data)

        # Send Email with Test Code Only
        send_test_email(user_email, test_code, topic, date, start, end, marks, level)

        flash('Test saved successfully and email sent!', 'success')
        return redirect(url_for('dashboard'))

    except Exception as e:
        print("Error while saving test:", str(e))
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for('questions_generation'))


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_test_email(to_email, test_code, topic, date, start, end, marks, level):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = "Your Test Details"

        # Email Body (No Questions, Only Test Code & Details)
        body = f"""
        <h2>Test Details</h2>
        <p><strong>Topic:</strong> {topic}</p>
        <p><strong>Date:</strong> {date}</p>
        <p><strong>Time:</strong> {start} - {end}</p>
        <p><strong>Marks:</strong> {marks}</p>
        <p><strong>Level:</strong> {level}</p>
        <p><strong>Your Test Code:</strong></p>
        <pre style="background-color: #f4f4f4; padding: 10px; border-left: 5px solid #007bff; font-size: 18px;">
{test_code}
        </pre>
        <p>Use this test code to access your test.</p>
        """

        msg.attach(MIMEText(body, 'html'))

        # Send Email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        server.quit()

        print(f"✅ Email sent successfully to {to_email}")

    except Exception as e:
        print("❌ Error sending email:", str(e))

recording = False
samplerate = 44100  # Default sample rate
audio_queue = queue.Queue()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_on_mask(mask, side,shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

import gdown
file_id = "1vPH1b8UJWjzGsGqEx31AmDGVxgWzDYFm"
output_path = "models/shape_predictor_68_face_landmarks.dat"

os.makedirs("models", exist_ok=True)
def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        return cx, cy
    except:
        return None, None
    
if not os.path.exists(output_path):
    print("Downloading shape_predictor_68_face_landmarks.dat from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False, fuzzy=True)
    print("Download complete!")
else:
    print("File already exists. Skipping download.")

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(output_path)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
# Video capture
cap = cv2.VideoCapture(0)
kernel = np.ones((9, 9), np.uint8)

threshold = 113

direction_counter = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
look_direction = None
last_direction = None
direction_start_time = None
DIRECTION_HOLD_TIME = 5

# Function to process the video feed and detect face movement
def detect_face_movement(frame, username, test_code):
    global look_direction, direction_counter, last_direction, direction_start_time
    start = time.time()
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    img.flags.writeable = False
    results = face_mesh.process(img)
    
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = img.shape
    face_3d = []
    face_2d = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extracting relevant landmarks (nose, eyes, and mouth)
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            
            # Converting lists to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            # Defining the camera matrix and distortion coefficients
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            # SolvePnP expects the 3D and 2D points to be reshaped
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            # Rotational matrix from the rotation vector
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360
            
            # Determining the direction based on the rotation angles
            if y_angle < -5:
                text = "Looking left"
            elif y_angle > 5:
                text = "Looking right"
            elif x_angle < -10:
                text = "Looking down"
            elif x_angle > 10:
                text = "Looking up"
            else:
                text = "Forward"
            
            if text != "Forward":
                if last_direction == text:
                    elapsed_time = time.time() - direction_start_time
                    if elapsed_time >= DIRECTION_HOLD_TIME:
                        print(f"Taking snapshot due to prolonged {text} movement")
                        snapshot_path = take_snapshot(img, text)
                        direction_start_time = time.time()  # Reset timer
                else:
                    last_direction = text
                    direction_start_time = time.time()
            else:
                last_direction = None
                direction_start_time = None
            # Projecting 3D nose to 2D
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            
            # Drawing a line from the nose
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))
            
            # cv2.line(img, p1, p2, (255, 0, 0), 3)
            
            # Displaying the text and angles
            # cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            # cv2.putText(img, "x: " + str(np.round(x_angle, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(img, "y: " + str(np.round(y_angle, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(img, "z: " + str(np.round(z_angle, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate the frame-per-second (FPS) value
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        print("FPS: ", fps)
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        # Draw face landmarks
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,  
        #     landmark_drawing_spec=drawing_spec,
        #     connection_drawing_spec=drawing_spec
        # )
    
    # Display the image
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        
        # Create masks for left and right eyes
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left, shape)  # ✅ Pass shape explicitly
        mask = eye_on_mask(mask, right, shape)  # ✅ Pass shape explicitly
        mask = cv2.dilate(mask, kernel, 5)

        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]

        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)

        # Pupil positions
        left_pupil = contouring(thresh[:, 0:mid], mid, img)
        right_pupil = contouring(thresh[:, mid:], mid, img, True)

        # Gaze detection logic
        if left_pupil != (None, None) and right_pupil != (None, None):
            left_eye_center = np.mean(shape[left], axis=0).astype("int")
            right_eye_center = np.mean(shape[right], axis=0).astype("int")

            # if left_pupil[0] < left_eye_center[0] and right_pupil[0] < right_eye_center[0]:
            #     print("Looking Right")
            #     cv2.putText(img, "Looking Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # elif left_pupil[0] > left_eye_center[0] and right_pupil[0] > right_eye_center[0]:
            #     print("Looking Left")
            #     cv2.putText(img, "Looking Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # else:
            #     print("Looking Center")
            #     cv2.putText(img, "Looking Center", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image
    # cv2.imshow('eyes', img)

    ret, frame = cap.read()

        # Perform object detection
    results = model(frame)  # Returns a list of Results objects

    phone_detected = False
    obstacle_detected = False

        # Access the first result object
    result = results[0]

        # Check if there are any detected objects
    if result.boxes is not None:
            for box in result.boxes:  # Iterate through detected objects
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class ID
                label = model.names[cls]  # Object label

                # Check if detected object is a mobile phone
                if label == "cell phone":
                    phone_detected = True
                    text_label = "Phone Detected"
                    color = (0, 255, 0)  # Green

                # If something is in front of the camera (possible obstacles)
                elif label in ["hand", "bottle", "cup", "remote", "book"]:
                    obstacle_detected = True
                    text_label = "Obstacle Detected"
                    color = (0, 0, 255)  # Red

                else:
                    continue

                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display messages on screen
    if phone_detected:
            cv2.putText(frame, "Phone Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    if obstacle_detected:
            cv2.putText(frame, "Obstacle Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
    if phone_detected or obstacle_detected:
        print("Taking snapshot due to detected object")
        snapshot_path = take_snapshot(frame, "object_detected")

        # Retrieve test code safely

        # Get user email from Firestore
        user_ref = db.collection('users').document(username).get()
        user_data = user_ref.to_dict() if user_ref.exists else None

        if not user_data or 'email' not in user_data:
            flash("User email not found!", "danger")
            return frame  # Don't break the stream

        user_email = user_data['email']
        

        # Send alert email
        send_alert_email(user_email, test_code, snapshot_path)
    return img





from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage  # Add this import

def send_alert_email(to_email, test_code, snapshot_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = "🚨 Alert: Unauthorized Object Detected!"  # Emoji in subject

        # Email Body (using UTF-8 encoding)
        body = f"""
        <h2 style="color: red;">🚨 Alert: Unauthorized Object Detected!</h2>
        <p><strong>Test Code:</strong> <pre>{test_code}</pre></p>
        <p>A mobile phone or unauthorized object was detected during your test.</p>
        <p>Please review the attached snapshot.</p>
        """

        msg.attach(MIMEText(body, 'html', 'utf-8'))  # Ensure UTF-8 encoding

        # Attach snapshot image
        with open(snapshot_path, 'rb') as f:
            img_data = f.read()
        image_attachment = MIMEImage(img_data, name="snapshot.jpg")
        msg.attach(image_attachment)

        # Send Email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string().encode('utf-8'))  # UTF-8 Encoding
        server.quit()

        print(f"✅ Alert email sent successfully to {to_email}")

    except Exception as e:
        print(f"❌ Error sending alert email: {str(e)}")
# def take_snapshot(image, direction):
#     if not os.path.exists('snapshots'):
#         os.makedirs('snapshots')
    
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     filename = f"snapshots/snapshot_{direction}_{timestamp}.jpg"
#     cv2.imwrite(filename, image)

def gen_frames(username, test_code):
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_face_movement(frame, username, test_code)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

model1 = whisper.load_model("base")

samplerate = 16000  
channels = 1
blocksize = 1024
dtype = 'float32'

audio_queue = queue.Queue()
recording = False

@app.route("/get_questions", methods=["POST"])
def get_questions():
    data = request.json
    print("Received request data:", data)  # Debug print

    input_code = data.get("input_code")
    if not input_code:
        print("Missing input code")  # Debug
        return jsonify({"error": "Missing input code"}), 400

    doc_ref = db.collection("tests").document(input_code)
    doc = doc_ref.get()

    if not doc.exists:
        print(f"Test code {input_code} not found in Firebase")  # Debug
        return jsonify({"error": "Test not found"}), 404

    questions_data = doc.to_dict().get("questions", [])
    test_data = doc.to_dict()
    # Ensure marks are included in the response
    questions = [
        {"question": q.get("question", ""), "marks": q.get("marks", 0)}
        for q in questions_data
    ]
    session["test_code"] = input_code
    session["topic"] = test_data.get("topic", "Unknown Topic") 
    print("Fetched questions with marks:", questions)  # Debug

    return jsonify({"questions": questions})


def record_audio():
    global recording
    recording = True
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        while recording:
            sd.sleep(100)

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Start recording
@app.route("/start", methods=["POST"])
def start_recording():
    global recording
    if not recording:
        threading.Thread(target=record_audio, daemon=True).start()
    return jsonify({"status": "Recording started"})

@app.route("/stop", methods=["POST"])
def stop_recording():
    global recording
    recording = False
    
    audio_data = []
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())
    
    if not audio_data:
        return jsonify({"error": "No audio recorded"})

    audio_np = np.concatenate(audio_data, axis=0).flatten()
    audio_np = audio_np / np.max(np.abs(audio_np))
    
    if samplerate != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=samplerate, target_sr=16000)
    
    model_whisper = whisper.load_model("base")
    result = model_whisper.transcribe(audio_np)
    transcription = result["text"]
    
    return jsonify({"transcription": transcription})

# Check answer with Gemini API
@app.route("/check_answer", methods=["POST"])
def check_answer():           
    if 'username' not in session:
        flash('Please log in first!', 'danger')
        return redirect(url_for('index'))

    username = session.get("username", "Unknown User")  # Get username safely
    email = session.get("email", "default_email@example.com")  # Retrieve email

    
    
    data = request.json
    user_answer = data.get("answer", "")
    question = data.get("question", "")
    total_marks = data.get("marks", 0)

    if not user_answer or not question:
        return jsonify({"error": "Missing question or answer"}), 400

    prompt = (
        f"Is the following answer correct or partially correct?\n"
        f"Question: {question}\nAnswer: {user_answer}\n"
        f"Provide a detailed evaluation. In case of any spelling mistake, please neglect it. please dont be strict be lenient"
        f"The question is for {total_marks} marks. Provide the awarded marks in bold, like this: **3** out of {total_marks}."
    )

    response = model2.generate_content(prompt)
    evaluation = response.text

    prompt2 = (
        f"Question: {question}\n"
        f"Maximum marks: {total_marks}\n"
        f"User's Answer: {user_answer}\n"
        f"Evaluate the answer strictly based on correctness, but ignore minor spelling mistakes.\n"
        f"Assign marks between 0 and {total_marks} based on the accuracy of the response.\n"
        f"Only return the awarded marks as a number, nothing else."
    )
    evaluation_marks = model2.generate_content(prompt2)
    evaluation_mark = evaluation_marks.text
    
    print("AI RESPONSE:", evaluation)  # Debugging: Check AI response format

    # Extract awarded marks using regex (handle multiple formats)

    # Save awarded marks to Firebase
    doc_ref = db.collection("evaluations").document()
    doc_ref.set({
        "question": question,
        "user_answer": user_answer,
        "evaluation": evaluation,
        "total_marks": total_marks,
        "awarded_marks": evaluation_mark
    })
    test_code = session.get("test_code", "Unknown Test")
    topic = session.get("topic", "Unknown Topic")
    print("AI MARKS RESPONSE:", evaluation_mark)
    send_marks_email(username, email, test_code, topic, evaluation_mark, total_marks)


    return jsonify({"evaluation": evaluation, "awarded_marks": evaluation_mark})

def send_marks_email(username, user_email, test_code, topic, awarded_marks, total_marks):
    subject = f"Quiz Results - {test_code}"
    
    email_body = f"""Hello {username},

Your marks on {test_code} ({topic}):
Total Marks Awarded: {awarded_marks}
Total Marks: {total_marks}

Thank you!
"""
    # Setup Email
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = user_email
    msg["Subject"] = subject

    msg.attach(MIMEText(email_body, "plain"))

    # Connect to SMTP Server
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Secure the connection
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, user_email, msg.as_string())
        
        print(f"Email sent successfully to {user_email}!")
    
    except Exception as e:
        print(f"Failed to send email: {str(e)}")


@app.route('/join_test')
def join_test():
    if 'username' not in session:
        flash('Please log in first!', 'danger')
        return redirect(url_for('index'))
    return render_template('join_test.html')

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return "Unauthorized", 401
    username = session['username']
    test_code = session.get('test_code', 'Unknown Test')
    return Response(gen_frames(username, test_code), mimetype='multipart/x-mixed-replace; boundary=frame')

def take_snapshot(image, direction):
    if not os.path.exists('snapshots'):
        os.makedirs('snapshots')
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"snapshots/snapshot_{direction}_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    return filename

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
