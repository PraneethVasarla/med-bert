from dataset import CORD_Dataset
from src import CustomNERModel
import configparser

config = configparser.ConfigParser()
config.read("config/project_config.ini")

push_to_hub = config['huggingface']['PUSH_TO_HUB']
dataset_path = config['dataset']['DATASET_PATH_CORD']

dataset = CORD_Dataset(dataset_path=dataset_path)
print(dataset.df.head())

model_checkpoint = config['huggingface']['MODEL_CHECKPOINT']

model_trainer = CustomNERModel(dataset=dataset,model_checkpoint=model_checkpoint)

# model_trainer = CustomNERModel(dataset=dataset,model_checkpoint=model_checkpoint)

model_trainer.train(model_name="bio-bert",push_to_hub=push_to_hub,all_rows=False)
