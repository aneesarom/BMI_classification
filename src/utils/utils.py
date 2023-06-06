import os
import sys
import pickle
from src.exception.exception import CustomException


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
    except Exception as err:
        raise CustomException(sys, err)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as err:
        raise CustomException(sys, err)