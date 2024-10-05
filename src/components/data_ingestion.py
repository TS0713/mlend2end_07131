import os
import sys
from src import src_dir
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.config import DataIngestionConfig


class DataIngestion:
    
    def __init__(self):
         self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered Data Ingestion Method or Component")
        try:
            raw_data_path_test = os.path.join(src_dir,"notebook","data","stud.csv")
            df = pd.read_csv(raw_data_path_test)
            logging.info("Read the dataset as data frame")

            os.makedirs(os.path.join(src_dir,"artifacts"),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train Test Split Initiated")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            logging.info("Sucessfull - Splitting Raw Data")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info(f"Sucessfull - Saving Train Data file: {os.path.basename(self.ingestion_config.train_data_path)} in the location: {os.path.dirname(self.ingestion_config.train_data_path)}")

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"Sucessfull - Saving Test Data file: {os.path.basename(self.ingestion_config.test_data_path)} in the location: {os.path.dirname(self.ingestion_config.test_data_path)}")

            logging.info("Data Ingestion - Successfull")
            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info(str(e))
            raise CustomException(str(e),sys)
            

    
'''
if __name__=="__main__":
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()

'''


