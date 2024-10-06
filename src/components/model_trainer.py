import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model
from src.components.config import ModelTrainerConfig, DataTransformationConfig, ModelList


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:

            preprocessor_path = DataTransformationConfig.preprocessor_obj_file_path

            logging.info("Splitting training and testing data")
            x_train,y_train,x_test,y_test = (
                                            train_array[:,:-1],
                                            train_array[:,-1],
                                            test_array[:,:-1],
                                            test_array[:,-1]
                                             )
            

            model_report = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=ModelList.models)

            # Best Model Score
            best_model_score = sorted(model_report.items(), key = lambda x: x[1], reverse=True)[0][1]

            # Best Model Name
            best_model_name = sorted(model_report.items(), key = lambda x: x[1], reverse=True)[0][0]

            # Best Model
            best_model = ModelList.models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(str("No Best Model Found"))
            
            logging.info("Best Model found")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
                )
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square
        except Exception as e:
            logging.info(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        

if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    #from src.components.model_trainer import ModelTrainer

    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_array, test_array = data_transformation_obj.initiate_data_transformation()

    model_trainer_obj = ModelTrainer()
    print (model_trainer_obj.initiate_model_trainer(train_array=train_array,test_array=test_array))



