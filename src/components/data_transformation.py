import os
import sys
import pandas as pd
import numpy as np
from src.logging.logging import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_pickle_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC',
                        'CALC', 'MTRANS']
            num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            target_col = ['NObeyesdad']

            target_categories = ['Obesity_Type_I', 'Obesity_Type_III', 'Obesity_Type_II',
                                 'Overweight_Level_I', 'Overweight_Level_II', 'Normal_Weight',
                                 'Insufficient_Weight']

            # create pipeline to impute missing data and encode the columns to numerical
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder())
            ])

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ])

            target_pipeline = Pipeline(steps=[
                ("encode", OrdinalEncoder(categories=[target_categories]))
            ])

            preprocessor = ColumnTransformer([
                ("cat", cat_pipeline, cat_cols),
                ("num", num_pipeline, num_cols)
            ])

            target_preprocessor = ColumnTransformer([
                ("target", target_pipeline, target_col)
            ])

            logging.info("Preprocessor file has been successfully created")
            return preprocessor, target_preprocessor

        except Exception as err:
            raise CustomException(sys, err)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation has been initiated")

            # read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train dataframe head: \n {train_df.head().to_string()}")
            logging.info(f"Test dataframe head: \n {test_df.head().to_string()}")

            target_feature_col = "NObeyesdad"

            # split the data into input and target feature
            input_feature_train_df = train_df.drop([target_feature_col], axis=1)
            target_feature_train = train_df[target_feature_col]
            input_feature_test_df = test_df.drop([target_feature_col], axis=1)
            target_feature_test = test_df[target_feature_col]
            logging.info("Input and Target feature has been successfully split")

            # transform the feature using preprocessor
            preprocessor_obj, target_preprocessor_obj = self.get_data_transformation_object()
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_test_arr = preprocessor_obj.transform(input_feature_test_df)
            target_feature_train = target_preprocessor_obj.fit_transform(pd.DataFrame(target_feature_train))
            target_feature_test = target_preprocessor_obj.transform(pd.DataFrame(target_feature_test))

            train_arr = np.c_[input_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_test_arr, np.array(target_feature_test)]

            logging.info("Data Transformation has been successfully completed")

            save_object(self.data_transformation_config.preprocessor_pickle_path, preprocessor_obj)
            logging.info("Preprocessor picked was successfully saved")

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_pickle_path)
        except Exception as err:
            raise CustomException(sys, err)
