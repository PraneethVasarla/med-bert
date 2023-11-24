from dataset import NERDataset
from src import CustomNERModel
import configparser

config = configparser.ConfigParser()
config.read("config/project_config.ini")

push_to_hub = config['huggingface']['PUSH_TO_HUB']
dataset_path = config['dataset']['DATASET_PATH']

dataset = NERDataset(dataset_path=dataset_path)
print(dataset)

model_checkpoint = config['huggingface']['MODEL_CHECKPOINT']

model_trainer = CustomNERModel(dataset=dataset,model_checkpoint=model_checkpoint)

model_trainer.train(model_name="med-bert",push_to_hub=push_to_hub)
