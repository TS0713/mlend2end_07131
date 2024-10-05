import os
from src import src_dir
from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.components.config import DataTransformationConfig, DataIngestionConfig
from src.components.config import DataFields
from src.utils import save_object



class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "median")),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Numercal Column Standard Scaling completed")

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Categorical Column encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,DataFields.numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,DataFields.categorical_columns)
                ]
            )

            logging.info("Combine numerical & categorical pipeline into single pipeline")

            return preprocessor

        except Exception as e:
            logging(str(e))
            raise CustomException(str(e),sys)
        
    def initiate_data_transformation(self):
        try:

            train_df = pd.read_csv(DataIngestionConfig.train_data_path)
            test_df = pd.read_csv(DataIngestionConfig.test_data_path)

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[DataFields.target_column],axis=1)
            target_feature_train_df = train_df[DataFields.target_column]

            input_feature_test_df = test_df.drop(columns=[DataFields.target_column],axis=1)
            target_feature_test_df = test_df[DataFields.target_column]

            logging.info("Started preprocessing on Train & Test Datasets")

            input_feature_train_processed_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_processed_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Sucessfully done preprocessing on Train & Test Datasets")

            train_arr = np.c_[
                input_feature_train_processed_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_processed_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path = DataTransformationConfig.preprocessor_obj_file_path,
                obj = preprocessing_obj
                )
            
            return (
                train_arr,
                test_arr
            )
        
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)





if __name__ == "__main__":
    from src.components.config import DataIngestionConfig, DataTransformationConfig, DataFields
    from src.components.data_ingestion import DataIngestion
    #from src.components.data_transformation import DataTransformation

    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    data_transformation_obj.initiate_data_transformation()





    

