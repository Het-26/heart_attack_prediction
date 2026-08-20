import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart_attack_prediction_dataset.csv')
CLEAN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_attack_pipeline.pkl')
INPUT_JSON_PATH = os.path.join(BASE_DIR, 'inputs', 'patient.json')

TARGET_COLUMN = 'Heart Attack Risk'

DROP_COLUMNS = ['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Income']

NUMERIC_FEATURES = [
    'Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week',
    'Stress Level', 'Sedentary Hours Per Day', 'BMI', 'Triglycerides',
    'Physical Activity Days Per Week', 'Sleep Hours Per Day',
    'Systolic', 'Diastolic'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
    'Alcohol Consumption', 'Diet', 'Previous Heart Problems',
    'Medication Use'
]
