import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def load_data(self, file_path: str):
        """ Load data from a CSV file """ 
        self.data = pd.read_csv(file_path)

    def clean_data(self):
        """ Clean the data by handling missing values and duplicates """
        self.data = self.data.drop_duplicates()
        self.data = self.data.fillna(self.data.mean())  # Fill missing values with mean

    def feature_engineering(self):
        """ Create new features based on the existing ones """
        self.data['new_feature'] = self.data['existing_feature'] ** 2  # Example feature

    def normalize_data(self):
        """ Normalize the data """
        scaler = StandardScaler()
        self.data[self.data.columns] = scaler.fit_transform(self.data[self.data.columns])

    def get_processed_data(self):
        """ Return the processed data """ 
        return self.data

    def split_data(self, target_column: str):
        """ Split the data into training and testing sets """ 
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)