import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import dataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig: # where to save train, test, raw data paths
    train_data_path = os.path.join('artefacts','train.csv')
    test_data_path = os.path.join('artefacts','test.csv')
    data_path = os.path.join('artefacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion process Started")
        
        try:
            dataFr = pd.read_csv('data/StudentsPerformance.csv')
            logging.info("Reading Data Set as DataFrame")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            dataFr.to_csv(self.ingestion_config.data_path, index=False, header=True)
            
            logging.info('Data Split Started')
            
            train_data,test_data = train_test_split(dataFr, test_size=0.2, random_state=42)
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data Ingestion Completed')
            
            # will be used for dataTransformation
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
        except Exception as e:
            raise CustomException(e,sys)
    
    
if __name__=='__main__':
    
    obj=DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    
    
    data_transformation = dataTransformation()
    train_arr, test_array,_ =  data_transformation.initiate_data_transformation(train_path, test_path)
    
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_trainer(train_arr, test_array )
    
    print('r2_score',r2_score)