import os
import csv
import io
import json
import re
from datetime import datetime, timedelta
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def get_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pro_secret_key_99'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

raw_db_url = os.getenv('DATABASE_URL')
if raw_db_url and raw_db_url.startswith('postgres://'):
    raw_db_url = raw_db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = raw_db_url if raw_db_url else 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_timeout': 20
}

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from models import db, Patient, Doctor, Admin, ScanReport, ChatHistory, DiseaseType
db.init_app(app)

login_manager = LoginManager(app)
migrate = Migrate(app, db)
login_manager.login_view = 'patient_login'

@login_manager.user_loader
def load_user(user_id):
    user_type = session.get('user_type')
    
    if user_type == 'admin':
        return Admin.query.get(int(user_id))
    elif user_type == 'patient':
        return Patient.query.get(int(user_id))
    else:
        user = Admin.query.get(int(user_id))
        if user:
            return user
        return Patient.query.get(int(user_id))

def login_admin(admin):
    session['user_type'] = 'admin'
    login_user(admin)

def login_patient(patient):
    session['user_type'] = 'patient'
    login_user(patient)

@app.route('/logout')
@login_required
def logout():
    session.pop('user_type', None)
    logout_user()
    return redirect(url_for('index'))

# LOAD THE NEW RESPIRATORY ANALYZER
print("Loading Respiratory Audio Analyzer...")
respiratory_analyzer = None
try:
    from ml_models.respiratory_analyzer import AcousticRespiratoryFramework
    respiratory_analyzer = AcousticRespiratoryFramework()
    print("Respiratory Analyzer loaded successfully")
except ImportError as e:
    print(f"Some ML dependencies missing: {e}")
    print("Running in basic mode without ML capabilities")
except Exception as e:
    print(f"Error loading respiratory analyzer: {e}")
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient-login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'POST':
        phone = request.form.get('phone')
        patient = Patient.query.filter_by(phone=phone).first()
        
        if patient:
            login_patient(patient)
            return redirect(url_for('patient_dashboard'))
        else:
            flash('Patient not found. Please contact admin.', 'danger')
    
    return render_template('patient_login.html')

@app.route('/patient-dashboard')
@login_required
def patient_dashboard():
    if not isinstance(current_user, Patient):
        flash('Access denied. Patient login required.', 'danger')
        return redirect(url_for('patient_login'))
    
    nearby_doctors = Doctor.query.filter_by(district=current_user.district).all()
    scan_reports = ScanReport.query.filter_by(patient_id=current_user.id).order_by(ScanReport.created_at.desc()).all()
    
    return render_template('patient_dashboard.html', 
                         patient=current_user, 
                         doctors=nearby_doctors,
                         scan_reports=scan_reports)

@app.route('/upload-scan', methods=['POST'])
@login_required
def upload_scan():
    if not isinstance(current_user, Patient):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'scan' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    file = request.files['scan']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{current_user.id}_{get_ist().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if respiratory_analyzer:
                results = respiratory_analyzer.analyze_audio(filepath)
            else:
                results = {
                    'disease_type': 'pending_analysis',
                    'disease_name': 'Analysis Pending',
                    'disease_description': 'ML model unavailable. Manual review required.',
                    'severity': 'Unknown',
                    'treatment_time': 'N/A',
                    'action_needed': 'Consult doctor',
                    'confidence': 0.5,
                    'acoustic_features': 'Analysis pending - ML model unavailable',
                    'recommendations': '<p class="text-warning">ML model unavailable. Please consult with a doctor for proper diagnosis.</p>',
                    'recommendations_raw': {},
                }
            
            recommendations_raw = results.get('recommendations_raw', {})
            if not isinstance(recommendations_raw, dict):
                recommendations_raw = {}
            
            report = ScanReport(
                patient_id=current_user.id,
                audio_path=filename, 
                disease_type=results.get('disease_type', 'unknown'),
                disease_name=results.get('disease_name', 'Unknown Condition'),
                disease_description=results.get('disease_description', 'No description available'),
                severity=results.get('severity', 'Unknown'),
                treatment_time=results.get('treatment_time', 'N/A'),
                action_needed=results.get('action_needed', 'N/A'), 
                confidence=results.get('confidence', 0.5),
                acoustic_features=results.get('acoustic_features', 'Area not specified'), 
                recommendations=results.get('recommendations', '<p>No recommendations available.</p>'),
                recommendations_raw=json.dumps(recommendations_raw),
                doctor_notes='',
                audio_metadata=json.dumps([]), 
                created_at=get_ist()
            )
            
            db.session.add(report)
            db.session.commit()
            db.session.refresh(report)
            
            flash('Audio uploaded and analyzed successfully!', 'success')
            return redirect(url_for('view_scan_report', report_id=report.id))
            
        except Exception as e:
            print(f"Error in upload_scan: {str(e)}")
            flash(f'Error saving audio: {str(e)}', 'danger')
            return redirect(url_for('patient_dashboard'))
    else:
        flash('Invalid file type. Please upload a valid audio file.', 'danger')
        return redirect(url_for('patient_dashboard'))

@app.route('/scan-report/<int:report_id>')
@login_required
def view_scan_report(report_id):
    report = ScanReport.query.get_or_404(report_id)
    
    if isinstance(current_user, Patient) and report.patient_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    try:
        if report.audio_metadata:
            report.audio_metadata = json.loads(report.audio_metadata)
        else:
            report.audio_metadata = []
    except Exception:
        report.audio_metadata = []
    
    try:
        if report.recommendations_raw:
            report.recommendations_raw = json.loads(report.recommendations_raw)
        else:
            report.recommendations_raw = {}
    except:
        report.recommendations_raw = {}
    
    return render_template('scan_report.html', report=report)

@app.route('/delete_report/<int:report_id>', methods=['POST'])
@login_required
def delete_report(report_id):
    report = ScanReport.query.get_or_404(report_id)
    
    if isinstance(current_user, Patient) and report.patient_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], report.audio_path)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Could not delete file: {e}")
        
    db.session.delete(report)
    db.session.commit()
    
    flash('Diagnostic report permanently deleted.', 'success')
    
    if isinstance(current_user, Admin):
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('patient_dashboard'))

@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot():
    data = request.json
    user_message = data.get('message', '').lower()
    
    clean_msg = re.sub(r'[^\w\s]', '', user_message)
    words = set(clean_msg.split())
    
    knowledge_base = [
        {
            "tags": {"asthma", "wheeze", "wheezing", "inhaler", "breath"},
            "response": "Asthma is a chronic condition causing airway inflammation and hyperresponsiveness. It often presents with high-pitched wheezing. Immediate treatment usually involves a rescue inhaler (like Albuterol)."
        },
        {
            "tags": {"copd", "bronchitis", "emphysema", "smoke", "smoking"},
            "response": "COPD (Chronic Obstructive Pulmonary Disease) includes emphysema and chronic bronchitis. It causes obstructed airflow. Management is lifelong and often involves long-acting bronchodilators and quitting smoking."
        },
        {
            "tags": {"pneumonia", "infection", "fever", "lung", "crackles"},
            "response": "Pneumonia is an infection that inflames the air sacs in one or both lungs, often filling them with fluid. Treatment usually requires antibiotics, rest, and close monitoring of oxygen levels."
        },
        {
            "tags": {"cough", "coughing", "mucus", "phlegm", "sputum"},
            "response": "A persistent cough with mucus can be a sign of Bronchial conditions, COPD, or Pneumonia. Staying hydrated helps thin the mucus. If the sputum is green/yellow or accompanied by a fever, consult a doctor."
        },
        {
            "tags": {"recovery", "heal", "time", "how", "long"},
            "response": "Recovery time varies greatly. Acute bronchitis or pneumonia may take 2-4 weeks to clear. However, conditions like Asthma and COPD are chronic and require lifelong management rather than a 'cure'."
        },
        {
            "tags": {"medicine", "medication", "pills", "antibiotics", "treatment"},
            "response": "Medication depends on the diagnosis. Pneumonia requires antibiotics. Asthma/COPD requires bronchodilators and inhaled corticosteroids. Always follow your physician's exact prescription."
        },
        {
            "tags": {"diet", "food", "eat", "nutrition"},
            "response": "A healthy diet supports lung function. Drink plenty of water to keep respiratory mucus thin. Avoid foods that cause acid reflux, as that can trigger asthma symptoms."
        },
        {
            "tags": {"sleep", "sleeping", "bed", "night"},
            "response": "If you have trouble breathing at night, try sleeping with your head elevated using extra pillows. For asthma and COPD, ensure your room is free of dust, pet dander, and other allergens."
        }
    ]
    
    best_match = None
    highest_score = 0
    
    for item in knowledge_base:
        score = len(words.intersection(item["tags"]))
        if score > highest_score:
            highest_score = score
            best_match = item["response"]
            
    if highest_score == 0:
        response = "I am a specialized clinical assistant for Respiratory Diseases. I can answer questions about Asthma, COPD, Pneumonia, Bronchial conditions, symptoms, and treatments. What would you like to know?"
    else:
        response = best_match
    
    chat = ChatHistory(
        patient_id=current_user.id if isinstance(current_user, Patient) else None,
        user_message=data.get('message', ''),
        bot_response=response
    )
    db.session.add(chat)
    db.session.commit()
    
    return jsonify({'response': response, 'timestamp': get_ist().strftime('%Y-%m-%d %H:%M')})

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin = Admin.query.filter_by(username=username).first()
        
        if admin:
            if admin.check_password(password):
                login_admin(admin)
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid credentials', 'danger')
        else:
            flash('Invalid credentials', 'danger')
    
    return render_template('admin_login.html')

@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    if not isinstance(current_user, Admin):
        flash('Access denied. Admin login required.', 'danger')
        return redirect(url_for('admin_login'))
    
    patients = Patient.query.all()
    doctors = Doctor.query.all()
    scan_reports = ScanReport.query.order_by(ScanReport.created_at.desc()).limit(10).all()
    
    return render_template('admin_dashboard.html', 
                         patients=patients, 
                         doctors=doctors,
                         scan_reports=scan_reports)

@app.route('/admin/patients')
@login_required
def manage_patients():
    if not isinstance(current_user, Admin):
        return redirect(url_for('admin_login'))
    
    patients = Patient.query.all()
    return render_template('manage_patients.html', patients=patients)

@app.route('/api/patient/<int:patient_id>', methods=['GET'])
@login_required
def get_patient(patient_id):
    if not isinstance(current_user, Admin):
        return jsonify({'error': 'Unauthorized'}), 401
    
    patient = Patient.query.get_or_404(patient_id)
    return jsonify({
        'id': patient.id,
        'name': patient.name,
        'phone': patient.phone,
        'email': patient.email,
        'address': patient.address,
        'district': patient.district,
        'medical_history': patient.medical_history
    })

@app.route('/admin/patients/add', methods=['POST'])
@login_required
def add_patient():
    if not isinstance(current_user, Admin):
        return jsonify({'error': 'Unauthorized'}), 401
    
    patient = Patient(
        name=request.form.get('name'),
        phone=request.form.get('phone'),
        email=request.form.get('email'),
        address=request.form.get('address'),
        district=request.form.get('district'),
        medical_history=request.form.get('medical_history')
    )
    
    db.session.add(patient)
    db.session.commit()
    
    flash('Patient added successfully', 'success')
    return redirect(url_for('manage_patients'))

@app.route('/edit_patient/<int:patient_id>', methods=['POST'])
@login_required
def edit_patient(patient_id):
    if not isinstance(current_user, Admin):
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('admin_dashboard'))
        
    patient = Patient.query.get_or_404(patient_id)
    
    patient.name = request.form.get('name', patient.name)
    patient.phone = request.form.get('phone', patient.phone)
    patient.email = request.form.get('email', patient.email)
    patient.address = request.form.get('address', patient.address)
    patient.district = request.form.get('district', patient.district)
    patient.medical_history = request.form.get('medical_history', patient.medical_history)
    
    try:
        db.session.commit()
        flash('Patient record updated successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating patient: {str(e)}', 'danger')
        
    return redirect(url_for('manage_patients'))

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
@login_required
def delete_patient(patient_id):
    if not isinstance(current_user, Admin):
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    patient = Patient.query.get_or_404(patient_id)
    
    ScanReport.query.filter_by(patient_id=patient_id).delete()
    ChatHistory.query.filter_by(patient_id=patient_id).delete()
    
    db.session.delete(patient)
    db.session.commit()
    
    flash('Patient record permanently deleted.', 'success')
    return redirect(url_for('manage_patients'))

@app.route('/admin/patients/import', methods=['POST'])
@login_required
def import_patients():
    if not isinstance(current_user, Admin):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('manage_patients'))
    
    file = request.files['file']
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
    else:
        flash('Please upload CSV or Excel file', 'danger')
        return redirect(url_for('manage_patients'))
    
    for _, row in df.iterrows():
        patient = Patient(
            name=row.get('name', ''),
            phone=str(row.get('phone', '')),
            email=row.get('email', ''),
            address=row.get('address', ''),
            district=row.get('district', ''),
            medical_history=row.get('medical_history', '')
        )
        db.session.add(patient)
    
    db.session.commit()
    flash(f'Imported {len(df)} patients successfully', 'success')
    return redirect(url_for('manage_patients'))

@app.route('/admin/patients/export')
@login_required
def export_patients():
    if not isinstance(current_user, Admin):
        return jsonify({'error': 'Unauthorized'}), 401
    
    patients = Patient.query.all()
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['ID', 'Name', 'Phone', 'Email', 'Address', 'District', 'Medical History'])
    
    for p in patients:
        writer.writerow([p.id, p.name, p.phone, p.email, p.address, p.district, p.medical_history])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='patients.csv'
    )

@app.route('/admin/doctors')
@login_required
def manage_doctors():
    if not isinstance(current_user, Admin):
        return redirect(url_for('admin_login'))
    
    doctors = Doctor.query.all()
    return render_template('manage_doctors.html', doctors=doctors)

@app.route('/admin/doctors/add', methods=['POST'])
@login_required
def add_doctor():
    if not isinstance(current_user, Admin):
        return jsonify({'error': 'Unauthorized'}), 401
    
    doctor = Doctor(
        name=request.form.get('name'),
        phone=request.form.get('phone'),
        email=request.form.get('email'),
        address=request.form.get('address'),
        district=request.form.get('district'),
        specialization=request.form.get('specialization', 'Pulmonologist')
    )
    
    db.session.add(doctor)
    db.session.commit()
    
    flash('Doctor added successfully', 'success')
    return redirect(url_for('manage_doctors'))

@app.route('/admin/doctors/import', methods=['POST'])
@login_required
def import_doctors():
    if not isinstance(current_user, Admin):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('manage_doctors'))
    
    file = request.files['file']
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
    else:
        flash('Please upload CSV or Excel file', 'danger')
        return redirect(url_for('manage_doctors'))
    
    for _, row in df.iterrows():
        doctor = Doctor(
            name=row.get('name', ''),
            phone=str(row.get('phone', '')),
            email=row.get('email', ''),
            address=row.get('address', ''),
            district=row.get('district', ''),
            specialization=row.get('specialization', 'Pulmonologist')
        )
        db.session.add(doctor)
    
    db.session.commit()
    flash(f'Imported {len(df)} doctors successfully', 'success')
    return redirect(url_for('manage_doctors'))

@app.route('/admin/doctors/export')
@login_required
def export_doctors():
    if not isinstance(current_user, Admin):
        return jsonify({'error': 'Unauthorized'}), 401
    
    doctors = Doctor.query.all()
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['ID', 'Name', 'Phone', 'Email', 'Address', 'District', 'Specialization'])
    
    for d in doctors:
        writer.writerow([d.id, d.name, d.phone, d.email, d.address, d.district, d.specialization])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='doctors.csv'
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables created")
        
        admin = Admin.query.filter_by(username='admin').first()
        if not admin:
            admin = Admin(username='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Default admin created (username: admin, password: admin123)")
        else:
            if not admin.check_password('admin123'):
                admin.set_password('admin123')
                db.session.commit()
    
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)