import os
import numpy as np
import pandas as pd
import librosa
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

CSV_PATH = "datasets/patient_diagnosis.csv"
AUDIO_DIR = "datasets/audio_and_txt_files"
MODEL_SAVE_PATH = "ml_models/model.pkl"

LABEL_MAPPING = {
    'Asthma': 0,
    'Bronchiectasis': 1,
    'Bronchiolitis': 1,
    'URTI': 1,
    'LRTI': 1,
    'COPD': 2,
    'Healthy': 3,
    'Pneumonia': 4
}

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=5.0)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception:
        return None

def train_model():
    print("Loading dataset labels...")
    try:
        diagnosis_df = pd.read_csv(CSV_PATH, header=None, names=['patient_id', 'diagnosis'])
        diagnosis_dict = dict(zip(diagnosis_df['patient_id'].astype(str), diagnosis_df['diagnosis']))
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_PATH}")
        return

    X = []
    y = []

    print("Extracting acoustic features from audio files...")
    
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Could not find {AUDIO_DIR}")
        return
        
    valid_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    total_files = len(valid_files)
    
    if total_files == 0:
        print(f"No .wav files found in {AUDIO_DIR}")
        return

    for i, file in enumerate(valid_files):
        patient_id = file.split('_')[0]
        
        if patient_id in diagnosis_dict:
            raw_label = diagnosis_dict[patient_id]
            
            if raw_label in LABEL_MAPPING:
                numeric_label = LABEL_MAPPING[raw_label]
                file_path = os.path.join(AUDIO_DIR, file)
                
                features = extract_features(file_path)
                
                if features is not None:
                    X.append(features)
                    y.append(numeric_label)
                    
        if (i + 1) % 50 == 0 or (i + 1) == total_files:
            print(f"Processed {i + 1}/{total_files} files...")

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("No valid features could be extracted. Please check your audio files.")
        return

    print(f"\nFeature extraction complete. Total valid samples: {len(X)}")
    print("Splitting dataset into training and testing sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
    
    target_names = ['Asthma', 'Bronchial', 'COPD', 'Healthy', 'Pneumonia']
    present_classes = np.unique(y_test)
    present_target_names = [target_names[i] for i in present_classes]
    
    print(classification_report(y_test, y_pred, target_names=present_target_names))

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
    