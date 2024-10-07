from dataclasses import dataclass
import os
from src import src_dir


from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
    )

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(src_dir,"artifacts","train_data.csv")
    test_data_path: str = os.path.join(src_dir,"artifacts","test_data.csv")
    raw_data_path: str = os.path.join(src_dir,"artifacts","raw_data.csv")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(src_dir,"artifacts","preprocessor.pkl")

@dataclass
class DataFields:
    numerical_columns = ["writing_score","reading_score"]
    categorical_columns = ["gender",
                           "race_ethnicity",
                           "parental_level_of_education",
                            "lunch",
                            "test_preparation_course"
                                    ]
    target_column = "math_score"


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(src_dir,"artifacts","model.pkl")


@dataclass
class ModelList:
    models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
    
    params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
