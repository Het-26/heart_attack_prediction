import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.config import (
    CLEAN_DATA_PATH, MODEL_PATH, TARGET_COLUMN,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES
)


class HeartAttackPipeline:
    def __init__(self, clean_path=CLEAN_DATA_PATH, model_path=MODEL_PATH):
        self.clean_path = clean_path
        self.model_path = model_path
        self.pipeline = None

    def load_data(self):
        df = pd.read_csv(self.clean_path)
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        return X, y

    def build_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERIC_FEATURES),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),
                 CATEGORICAL_FEATURES),
            ],
            remainder='passthrough'
        )

        classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        # SMOTE lives inside the pipeline so it only runs during .fit()
        # on the training fold — no leakage into test data.
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', classifier),
        ])
        return self.pipeline

    def train_and_evaluate(self):
        X, y = self.load_data()

        # Split FIRST — SMOTE runs only on train fold inside the pipeline.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)

        print('\n=== Evaluation on Test Set ===')
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))

        return self.pipeline

    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f'Pipeline saved to {self.model_path}')

    def run(self):
        self.train_and_evaluate()
        self.save()
        return self.pipeline
