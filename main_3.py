from dataset import CORD_Dataset
from src import CutsomBertModel
import configparser

config = configparser.ConfigParser()
config.read("config/project_config.ini")

push_to_hub = bool(config['huggingface']['PUSH_TO_HUB'])
dataset_path = config['dataset']['DATASET_PATH_CORD']

dataset = CORD_Dataset(dataset_path=dataset_path)
print(dataset.df.head())

model_checkpoint = config['huggingface']['MODEL_CHECKPOINT']

model = CutsomBertModel(dataset=dataset,model_checkpoint=model_checkpoint)

model.train(model_name="finetuned-roberta-accelerate-ner",push_to_hub=push_to_hub)
