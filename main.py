import os
import argparse

from src.config import MODEL_PATH, CLEAN_DATA_PATH
from src.data_cleaning import DataCleaner
from src.train_pipeline import HeartAttackPipeline
from src.predictor import HeartAttackPredictor


def train():
    if not os.path.exists(CLEAN_DATA_PATH):
        print('Cleaned data not found. Running data cleaning...')
        DataCleaner().clean()
    else:
        print(f'Using existing cleaned data at {CLEAN_DATA_PATH}')

    print('\nTraining pipeline...')
    HeartAttackPipeline().run()


def main():
    parser = argparse.ArgumentParser(description='Heart Attack Risk Prediction')
    parser.add_argument(
        '--retrain', action='store_true',
        help='Force retraining even if a saved model exists'
    )
    args = parser.parse_args()

    if args.retrain or not os.path.exists(MODEL_PATH):
        if args.retrain:
            print('Retraining requested.')
        else:
            print(f'No saved model at {MODEL_PATH}. Training a new one.')
        train()
    else:
        print(f'Loading existing model from {MODEL_PATH}')

    HeartAttackPredictor().predict()


if __name__ == '__main__':
    main()
