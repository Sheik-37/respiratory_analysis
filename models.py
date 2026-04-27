from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Patient(UserMixin, db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(100))
    address = db.Column(db.String(200))
    district = db.Column(db.String(100))
    medical_history = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    scan_reports = db.relationship('ScanReport', backref='patient', lazy=True)
    chat_history = db.relationship('ChatHistory', backref='patient', lazy=True)
    
    def get_id(self):
        return str(self.id)

class Doctor(db.Model):
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100))
    address = db.Column(db.String(200))
    district = db.Column(db.String(100))
    specialization = db.Column(db.String(100), default='Pulmonologist')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Admin(UserMixin, db.Model):
    __tablename__ = 'admins'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_id(self):
        return str(self.id)

class DiseaseType(db.Model):
    __tablename__ = 'disease_types'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    treatment_guidelines = db.Column(db.Text)
    
    scan_reports = db.relationship('ScanReport', backref='disease_type_ref', lazy=True)

class ScanReport(db.Model):
    __tablename__ = 'scan_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    disease_type_id = db.Column(db.Integer, db.ForeignKey('disease_types.id'))
    audio_path = db.Column(db.String(200), nullable=False)
    
    disease_type = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    acoustic_features = db.Column(db.String(200))
    
    disease_name = db.Column(db.String(200))
    disease_description = db.Column(db.Text)
    severity = db.Column(db.String(50))
    treatment_time = db.Column(db.String(100))
    action_needed = db.Column(db.String(200))
    
    recommendations = db.Column(db.Text)
    recommendations_raw = db.Column(db.Text)
    doctor_notes = db.Column(db.Text)
    
    audio_metadata = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_name': self.patient.name if self.patient else 'Unknown',
            'disease_type': self.disease_type,
            'disease_name': self.disease_name,
            'severity': self.severity,
            'confidence': f"{self.confidence:.2%}" if self.confidence else 'N/A',
            'acoustic_features': self.acoustic_features,
            'date': self.created_at.strftime('%Y-%m-%d %H:%M')
        }

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'))
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)