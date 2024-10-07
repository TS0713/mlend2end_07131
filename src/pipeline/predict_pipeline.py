import sys
import pandas as pd
import numpy as np
from src.components.config import ModelTrainerConfig, DataTransformationConfig
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        try:
            self.trained_model = load_object(ModelTrainerConfig.trained_model_file_path)
            logging.info("Successfull - Loaded Trained Model")
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
    def predict(self,scaled_features):
        try:
            model_predictions = self.trained_model.predict(scaled_features)
            logging.info("Successfull - Model Predictions done")
            return model_predictions
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            CustomException(str(e),sys)
        
class CustomData:
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int
                ):
        try:
            self.preprocessor = load_object(DataTransformationConfig.preprocessor_obj_file_path)
            logging.info("Successfull - Loaded Preprocessor Pickel file")
            self.gender = gender
            self.race_ethnicity = race_ethnicity
            self.parental_level_of_education = parental_level_of_education
            self.lunch = lunch
            self.test_preparation_course = test_preparation_course
            self.reading_score = reading_score
            self.writing_score = writing_score
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)

    def feature_processing(self):
        try:
            user_data_input = pd.DataFrame({
                    "gender": [self.gender],
                    "race_ethnicity": [self.race_ethnicity],
                    "parental_level_of_education": [self.parental_level_of_education],
                    "lunch": [self.lunch],
                    "test_preparation_course": [self.test_preparation_course],
                    "reading_score": [self.reading_score],
                    "writing_score": [self.writing_score],
                })
            
            '''
            user_data_input = pd.DataFrame({
                    "gender": ["male"],
                    "race_ethnicity": ["group B"],
                    "parental_level_of_education": ["bachelor's degree"],
                    "lunch": ["standard"],
                    "test_preparation_course": ["completed"],
                    "reading_score": [99],
                    "writing_score": [90],
                })
            '''
            
            scaled_features = self.preprocessor.transform(user_data_input)
            logging.info("Successfull - Applied transformation on features")
            return scaled_features
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)



