import os
import sys
import numpy as np
import pandas as pd
from src.exception import CostumException
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    """
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CostumException(e, sys)


def evaluate_models(X_train, y_train, models, X_test, y_test):
    """
    Evaluate multiple machine learning models and return their R2 scores.

    """
    try:
        report = {}

        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)

            # Predict on test data
            y_pred = model.predict(X_test)

            # Calculate R2 score
            r2_score_value = r2_score(y_test, y_pred)

            report[model_name] = r2_score_value

        return report
    except Exception as e:
        raise CostumException(e, sys)