import pandas as pd
import os
import sys
from src.exception import CostumException
from src.logger import logging
from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



class DataTransformation:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def initiate_data_transformation(self):

        try:
            logging.info("Starting data transformation process")
            train = pd.read_csv(self.train_path)
            test = pd.read_csv(self.test_path)

        except Exception as e:
            raise CostumException(e, sys)
