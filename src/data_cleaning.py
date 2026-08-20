import pandas as pd
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH, DROP_COLUMNS


class DataCleaner:
    def __init__(self, raw_path=RAW_DATA_PATH, clean_path=CLEAN_DATA_PATH):
        self.raw_path = raw_path
        self.clean_path = clean_path
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.raw_path)
        return self

    def split_blood_pressure(self):
        bp = self.df['Blood Pressure'].str.split('/', expand=True)
        self.df['Systolic'] = bp[0].astype(int)
        self.df['Diastolic'] = bp[1].astype(int)
        self.df = self.df.drop(columns=['Blood Pressure'])
        return self

    def drop_unused_columns(self):
        cols_to_drop = [c for c in DROP_COLUMNS if c in self.df.columns]
        self.df = self.df.drop(columns=cols_to_drop)
        return self

    def save(self):
        self.df.to_csv(self.clean_path, index=False)
        return self

    def clean(self):
        self.load().split_blood_pressure().drop_unused_columns().save()
        print(f'Cleaned data saved to {self.clean_path}')
        return self.df
