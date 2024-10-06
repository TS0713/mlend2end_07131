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
                "K-Neighbours Regression": KNeighborsRegressor(),
                "XGB-Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
