import os
import json
import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
try:
    from firebase_admin import auth, db
    FIREBASE_IMPORT = True
except ImportError:
    auth = None
    db = None
    FIREBASE_IMPORT = False
from dotenv import load_dotenv
from app.utils.firebase_config import initialize_firebase
from app.utils.gait_processor import process_video, save_debug_frames
from app.models.cnn_model import load_dementia_model, predict_risk
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

load_dotenv()

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = 'dementia_detection_secret_key'

def build_firebase_web_config():
    return {
        "apiKey": os.getenv("FIREBASE_API_KEY", "AIzaSyA_pP2tEBmN-290LZqsmPk8P1sZ8J9YGKE"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "jayaseelan-878c9.firebaseapp.com"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", "jayaseelan-878c9"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "jayaseelan-878c9.firebasestorage.app"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "437943094350"),
        "appId": os.getenv("FIREBASE_APP_ID", "1:437943094350:web:95ef4a279181bfb665471f"),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", "G-QFTE1L9H0Q")
    }

FIREBASE_WEB_CONFIG = build_firebase_web_config()

# Initialize Firebase
firebase_app = initialize_firebase()
FIREBASE_READY = firebase_app is not None

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, uid, email, role='patient'):
        self.id = uid
        self.email = email
        self.role = role

def verify_token_with_identity_toolkit(id_token):
    """Verifies ID token through Firebase Identity Toolkit when Admin SDK isn't configured."""
    api_key = FIREBASE_WEB_CONFIG.get("apiKey")
    if not api_key:
        return None

    endpoint = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={api_key}"
    payload = json.dumps({"idToken": id_token}).encode("utf-8")
    request_obj = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request_obj, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    users = data.get("users", [])
    if not users:
        return None

    user_info = users[0]
    uid = user_info.get("localId")
    if not uid:
        return None

    return {"uid": uid, "email": user_info.get("email", "")}

@login_manager.user_loader
def load_user(user_id):
    role = session.get('user_role', 'patient')
    if FIREBASE_READY:
        try:
            user = auth.get_user(user_id)
            return User(user.uid, user.email or session.get("user_email", ""), role)
        except Exception:
            return None

    # Fallback path for environments without Admin SDK key.
    email = session.get("user_email")
    if email:
        return User(user_id, email, role)

    return None

@app.route('/')
def index():
    if current_user.is_authenticated:
        if getattr(current_user, 'role', 'patient') == 'clinician':
            return redirect(url_for('clinician_dashboard'))
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        flash("Please use Firebase login from the form.")
    return render_template('login.html', firebase_config=FIREBASE_WEB_CONFIG)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        flash("Please use Firebase registration from the form.")
    return render_template('register.html', firebase_config=FIREBASE_WEB_CONFIG)

@app.route('/session_login', methods=['POST'])
def session_login():
    payload = request.get_json(silent=True) or {}
    id_token = payload.get('idToken')
    is_registering = payload.get('is_registering', False)
    payload_role = payload.get('role', 'patient')

    if not id_token:
        return jsonify({"success": False, "message": "Missing Firebase ID token."}), 400

    verified_user = None

    if FIREBASE_READY:
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token.get("uid")
            user_record = auth.get_user(uid)
            verified_user = {"uid": user_record.uid, "email": user_record.email or decoded_token.get("email", "")}
        except Exception:
            verified_user = None

    if verified_user is None:
        verified_user = verify_token_with_identity_toolkit(id_token)

    if not verified_user:
        return jsonify({"success": False, "message": "Invalid or expired Firebase token."}), 401

    uid = verified_user["uid"]
    email = verified_user.get("email", "")
    
    # Handle Role in Database
    user_role = 'patient'
    if FIREBASE_READY:
        try:
            user_ref = db.reference(f'users/{uid}')
            if is_registering:
                user_ref.set({
                    'email': email,
                    'role': payload_role
                })
                user_role = payload_role
            else:
                user_data = user_ref.get()
                if user_data and 'role' in user_data:
                    user_role = user_data['role']
        except Exception as e:
            print(f"Error handling role: {e}")

    session["user_email"] = email
    session["user_role"] = user_role
    login_user(User(uid, email, user_role))

    redirect_url = url_for("clinician_dashboard") if user_role == 'clinician' else url_for("dashboard")
    return jsonify({"success": True, "redirect": redirect_url})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop("user_email", None)
    return redirect(url_for('login'))

@app.route('/clinician_dashboard')
@login_required
def clinician_dashboard():
    if session.get('user_role') != 'clinician':
        flash("Unauthorized access. Clinician portal only.", "danger")
        return redirect(url_for('dashboard'))
        
    patients_data = []
    if FIREBASE_READY and db:
        try:
            users_ref = db.reference('users').get() or {}
            patient_emails = {uid: data.get('email', 'Unknown') for uid, data in users_ref.items() if data.get('role') == 'patient'}
            
            all_results = db.reference('results').get() or {}
            
            for uid, records in all_results.items():
                if uid in patient_emails:
                    latest_record = None
                    for key, record in records.items():
                        if not latest_record or record.get('timestamp', '') > latest_record.get('timestamp', ''):
                            latest_record = record
                    
                    if latest_record:
                        patients_data.append({
                            'id': uid,
                            'email': patient_emails[uid],
                            'latest_risk': latest_record.get('risk_score', 0),
                            'latest_level': latest_record.get('risk_level', 'Unknown'),
                            'last_assessment': latest_record.get('timestamp', '')[:19],
                            'task_type': latest_record.get('task_type', 'single')
                        })
            
            patients_data.sort(key=lambda x: x['latest_risk'], reverse=True)
        except Exception as e:
            print(f"Error loading clinician dashboard: {e}")

    return render_template('clinician_dashboard.html', user=current_user, patients=patients_data)

@app.route('/dashboard')
@login_required
def dashboard():
    recent_history = []
    overall_health = 0
    status_label = "No Data"
    if FIREBASE_READY and db:
        try:
            ref = db.reference(f'results/{current_user.id}')
            data = ref.get()
            if data:
                all_records = []
                for key, value in data.items():
                    all_records.append(value)
                    
                all_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
                if all_records:
                    avg_score = sum(item.get('risk_score', 0) for item in all_records) / len(all_records)
                    overall_health = round(100 - avg_score, 1) # Convert risk to health %
                    
                    if avg_score >= 65: status_label = "Critical Decline"
                    elif avg_score >= 40: status_label = "Monitoring Required"
                    else: status_label = "Stable Baseline"
                    
                recent_history = all_records[:3]  # Only show top 3
        except Exception as e:
            print(f"Error fetching recent history: {e}")
            
    return render_template('dashboard.html', user=current_user, recent_history=recent_history, overall_health=overall_health, status_label=status_label)

@app.route('/history')
@login_required
def history():
    user_id = current_user.id
    history_data = []
    try:
        if db:
            ref = db.reference(f'results/{user_id}')
            data = ref.get()
            if data:
                for key, value in data.items():
                    history_data.append(value)
                history_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    except Exception as e:
        print(f"Error fetching history: {e}")
        
    return render_template('history.html', user=current_user, history=history_data)

@app.route('/predict', methods=['POST'])
def predict():
    task_type = request.form.get('task_type')
    video_file = request.files.get('video')
    
    if not video_file:
        flash("No video uploaded")
        return redirect(url_for('dashboard'))
    
    video_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_path)
    
    # Process video and get walking quality metrics
    processed_frames, walking_quality = process_video(video_path, task_type=task_type)

    # Predict using CNN + walking quality
    model = load_dementia_model()
    prediction = predict_risk(model, processed_frames, task_type, walking_quality)
    
    # Generate Clinical Evaluation Summary
    eval_notes = []
    if walking_quality.get('step_symmetry', 100) < 70:
        eval_notes.append("Significant asymmetry detected in striding patterns.")
    else:
        eval_notes.append("Step symmetry is within normal bilateral baseline.")
        
    if walking_quality.get('velocity', 100) < 60:
        eval_notes.append("Patient exhibits highly reduced ambulatory cadence.")
        
    if task_type == 'dual':
        if prediction.get('level') in ['High Risk', 'Moderate Risk']:
            eval_notes.append("Severe cognitive-motor interference observed under dual-task cognitive load, strongly indicating neurological decline.")
        else:
            eval_notes.append("Patient successfully maintained gross motor stability during continuous cognitive dual-tasking.")
            
    if prediction.get('score', 0) >= 75:
        eval_notes.append("Immediate clinical follow-up and neuropsychological assessment is recommended.")
    elif prediction.get('score', 0) >= 40:
        eval_notes.append("Monitor patient longitudinally for further gait deterioration.")
        
    evaluation_summary = " ".join(eval_notes)
    
    # Store in Firebase DB (if initialized)
    try:
        ref = db.reference(f'results/{current_user.id}')
        ref.push({
            'task_type': task_type,
            'risk_score': prediction['score'],
            'risk_level': prediction['level'],
            'timestamp': str(datetime.datetime.now())
        })
    except:
        print("Firebase DB not initialized, skipping storage.")
    
    return render_template('results.html', 
                           prediction=prediction, 
                           task_type=task_type,
                           evaluation_summary=evaluation_summary)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
