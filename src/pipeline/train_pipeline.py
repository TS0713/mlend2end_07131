from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def train_the_model():
    print ("Data Ingestion Started")
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()
    print ("Data Ingestion Completed")

    print ("Data Transformation Started")
    data_transformation_obj = DataTransformation()
    train_array, test_array = data_transformation_obj.initiate_data_transformation()
    print ("Data Transformation Completed")
    
    print ("Model Training Started")
    model_trainer_obj = ModelTrainer()
    print (model_trainer_obj.initiate_model_trainer(train_array=train_array,test_array=test_array))
    print ("Model Training Completed")
    


if __name__=="__main__":
    train_the_model()