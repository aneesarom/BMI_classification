from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_transformation = DataTransformation()
    train_path, test_path = data_ingestion.initialize_data_ingestion()
    train_arr, test_arr, _ = data_transformation.initialize_data_transformation(train_path, test_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)
