import os 
import sys

import pickle
from pathlib import Path

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score


def save_object(file_path: Path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as f:
            pickle.dump(obj,f)
        logging.info(f"Successully saved {file_name} in the location {dir_path}")
        
    except Exception as e:
        logging.info(f"Failed -  saving {file_name} in the location {dir_path}")
        logging.info(str(e))
        CustomException(str(e),sys)


def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            train_model_score = r2_score(y_train,y_train_pred)
            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logging.info(CustomException(str(e)))
        raise CustomException(str(e),sys)
    