from src.constant import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.utils import save_obj
from src.utils import evaluate_model

class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def intitate_model_trainning(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (train_array[:, :-1], train_array[:, -1],
                                                test_array[:, :-1], test_array[:, -1])

            models = {
                "XGBRegressor": XGBRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "SVR": SVR()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            print(model_report)   
            # sorted score
            best_model_score = max(sorted(model_report.values()))
            # model name and model score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            # Save the best model as a pickle file (using helper function)
            save_obj(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

        except Exception as e:
            raise CustomException(e, sys)
