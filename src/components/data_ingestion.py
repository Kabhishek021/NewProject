from src.constant import *

from src.config.configuration import *

import os,sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation,DataTransformationConfig

from src.components.model_trainer import ModelTrainer

# Define configuration (1)  :define raw data pah/train file path / test file path 
class DataIngestionconfig:

    raw_data_path:str = RAW_FILE_PATH
    train_data_path :str =  TRAIN_FILE_PATH
    test_data_path:str  = TEST_FILE_PATH
        

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionconfig()


    def initiate_data_ingestion(self):
        try:
            # 1 read dataset 
            df=pd.read_csv(DATASET_PATH)

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok =True)
            #2 save to raw_data_path
            df.to_csv(self.data_ingestion_config.raw_data_path ,index =False)

            #3 split into train and test
            train_set ,test_set =train_test_split(df ,test_size=0.20 ,random_state =42)

            #4 afer split data and save into traindatapath and testdatapath 
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path ),exist_ok =True)
            train_set.to_csv(self.data_ingestion_config.train_data_path)



            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path ),exist_ok =True)
            test_set.to_csv(self.data_ingestion_config.test_data_path)

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException( e, sys)

# if __name__ =='__main__':
#     obj = DataIngestion()
#     train_data_path ,test_data_path=obj.initiate_data_ingestion()


#     data_transformation =DataTransformation()
#     #train_arr and test_arr calling hee
#     train_arr ,test_arr = data_transformation.inititate_data_transformation(train_data_path ,test_data_path)

#model trainning 
if __name__ =='__main__':
    obj = DataIngestion()
    train_data_path ,test_data_path=obj.initiate_data_ingestion()


    data_transformation =DataTransformation()
    #train_arr and test_arr calling hee
    train_arr ,test_arr = data_transformation.inititate_data_transformation(train_data_path ,test_data_path)

    # model trainer
    model_trainer = ModelTrainer()
    print(model_trainer.intitate_model_trainning(train_arr ,test_arr))  
