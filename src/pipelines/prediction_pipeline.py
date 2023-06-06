import logging
import sys
import os
import pandas as pd
from src.utils.utils import load_object


class CustomData:
    def __init__(self, gender, family_history_with_overweight, favc, caec, smoke, scc, calc, mtrans, age, height,
                 weight, fcvc, ncp, ch2o, faf, tue):
        self.gender = gender
        self.family_history_with_overweight = family_history_with_overweight
        self.favc = favc
        self.caec = caec
        self.smoke = smoke
        self.scc = scc
        self.calc = calc
        self.mtrans = mtrans
        self.age = age
        self.height = height
        self.weight = weight
        self.fcvc = fcvc
        self.ncp = ncp
        self.ch2o = ch2o
        self.faf = faf
        self.tue = tue

    def get_data_as_dataframe(self):
        data_dict = {
                "Gender": [self.gender],
                "family_history_with_overweight": [self.family_history_with_overweight],
                "FAVC": [self.favc],
                "CAEC": [self.caec],
                "SMOKE": [self.smoke],
                "SCC": [self.scc],
                "CALC": [self.calc],
                "MTRANS": [self.mtrans],
                "Age": [self.age],
                "Height": [self.height],
                "Weight": [self.weight],
                "FCVC": [self.fcvc],
                "NCP": [self.ncp],
                "CH2O": [self.ch2o],
                "FAF": [self.faf],
                "TUE": [self.tue]
            }

        df = pd.DataFrame(data_dict)
        return df


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        # Set the path for pickle file
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        model_path = os.path.join("artifacts", "best_model.pkl")
        # load the pickle file
        preprocessor_file = load_object(preprocessor_path)
        model_file = load_object(model_path)
        data = preprocessor_file.transform(data)
        logging.info("Data has been successfully transformed in prediction pipeline")
        # predict the classification and find its probability
        predicted = model_file.predict(data)
        logging.info("Data has been successfully predicted in prediction pipeline")
        return predicted
