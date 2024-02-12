import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_pkl, train_evaluate_performance

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    train_path = os.path.join("artefacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_trainer(self, train_array, test_array):
        
        try:
            logging.info('Split data to train/test')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_report : dict = train_evaluate_performance(X_train=X_train,y_train=y_train,
                                                       X_test=X_test,y_test=y_test,models=models)
               
            #best score
            best_r2_score = max(sorted(model_report.values()))
                    # print('best_r2_score',best_r2_score)
            #best model      
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_r2_score)
            ]      
            print('best_model_name',best_model_name)
            best_model = models[best_model_name]    
            
            logging.info('Founding best model done : {0}, with r2_score = {1}'.format(
               best_model_name,best_r2_score
            ))     
            
            save_pkl(file_path=self.model_trainer_config.train_path,
                     obj=best_model)
            
            preds = best_model.predict(X_test)
            best_score = r2_score(y_test,preds)   
            return best_score        
        
        
        except Exception as e:
            raise CustomException(e,sys)