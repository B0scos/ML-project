import os
import sys
import numpy as np
import pandas as pd
from src.exception import CostumException
import dill

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