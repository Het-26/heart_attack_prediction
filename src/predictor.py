import json
import pickle
import pandas as pd
from src.config import MODEL_PATH, INPUT_JSON_PATH


class HeartAttackPredictor:
    def __init__(self, model_path=MODEL_PATH, input_path=INPUT_JSON_PATH):
        self.model_path = model_path
        self.input_path = input_path
        self.pipeline = None

    def load_pipeline(self):
        with open(self.model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        return self

    def load_input(self):
        with open(self.input_path, 'r') as f:
            data = json.load(f)
        # Accept a single dict — same format as before.
        df = pd.DataFrame([data])
        return df

    def predict(self):
        if self.pipeline is None:
            self.load_pipeline()
        df = self.load_input()
        pred = self.pipeline.predict(df)[0]
        prob = self.pipeline.predict_proba(df)[0][1]

        print('\n=== Heart Attack Risk Prediction ===')
        print(f'Predicted Risk : {"HIGH" if pred == 1 else "LOW"} ({pred})')
        print(f'Risk Probability: {prob:.4f}')
        return pred, prob
