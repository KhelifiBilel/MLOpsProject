import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..utils import save_pkl

@dataclass
class dataTransformationConfig:
    preprocessor_path = os.path.join('artefacts','preprocessor_data.pkl')
    
class dataTransformation:
    def __init__(self) -> None:
        self.dataTransformationConfig = dataTransformationConfig()
        
    def get_data_transformer(self):
        try:
            num_features = ['reading_score','writing_score']
            cat_features = ['gender','race_ethnicity','parental_level_of_education',
                            'lunch','test_preparation_course']

            num_pipeline = Pipeline(
                steps=[("impute", SimpleImputer(strategy="median")),
                       ("scale", StandardScaler())]
            )
            logging.info("Numerical feature Scanling completed")
            
            cat_pipeline = Pipeline(
                steps=[("impute", SimpleImputer(strategy="most_frequent")),
                       ("Encode", OneHotEncoder()),
                       ("scale", StandardScaler(with_mean=False))]
            )
            logging.info("Categorical feature encoding completed")
            
            preprocessor = ColumnTransformer(
                [('num_pipeline',num_pipeline,num_features),
                 ('cat_pipeline',cat_pipeline,cat_features)
                ]
            )
            logging.info(f"Categorical features : {cat_features}")
            logging.info(f"Numerical features : {num_features}")
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test Data')
            
            preprocessor_obj = self.get_data_transformer()
            
            label = 'math_score'

            X_train_df = train_df.drop(columns=[label],axis=1)
            y_train_df = train_df[label]
            
            X_test_df = test_df.drop(columns=[label],axis=1)
            y_test_df = test_df[label]
            
            logging.info("Preprocessing for train/test dataframes")
            
            X_train_arr = preprocessor_obj.fit_transform(X_train_df)
            X_test_arr = preprocessor_obj.transform(X_test_df)
            
            train_arr = np.c_[X_train_arr,np.array(y_train_df)]
            test_arr = np.c_[X_test_arr,np.array(y_test_df)]
            
            logging.info("Saving preprocessing object")
            
            save_pkl(file_path=self.dataTransformationConfig.preprocessor_path,
                    obj=preprocessor_obj)
            
            return(
                train_arr,
                test_arr,
                self.dataTransformationConfig.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e, sys)
            