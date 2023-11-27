import os,sys

sys.path.append(os.getcwd())

from transformers import AutoTokenizer,AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments,Trainer
import evaluate
import numpy as np
import torch
from huggingface_hub import login
import configparser
config = configparser.ConfigParser()
config.read("config/project_config.ini")

class CustomNERModel:
    def __init__(self,dataset = None,model_checkpoint="bert-base-uncased") -> None:
        self.dataset = dataset
        self.dataset.prepare_hf_dataset()

        self.model_checkpoint = model_checkpoint
    
    def align_labels_with_tokens(self,labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                try:
                    label = -100 if word_id is None else labels[word_id]
                except:
                    print(f"Value of labels: {labels} and value of word_id: {word_id}, word_ids:{word_ids}")
                    break
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = int(labels[word_id])
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels
    
    def tokenize_and_align_lables(self,examples):
        tokenized_inputs = self.tokenizer(examples['sentence'],truncation=True,is_split_into_words=True)
        all_labels = examples['bio_tags']
        new_labels = []
        for i,labels in enumerate(all_labels):
            labels = [int(x) for x in labels]
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(word_ids=word_ids,labels=labels))

        tokenized_inputs['labels'] = new_labels
        return tokenized_inputs
    
    def prepare_data(self):
        self.tokenized_datasets = self.dataset.hf_dataset.map(
            self.tokenize_and_align_lables,batched=True,remove_columns=self.dataset.hf_dataset['train'].column_names
            )
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.label_names = list(self.dataset.label2id.keys())

    def compute_metrics(self,eval_preds):
        metric = evaluate.load("seqeval")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.dataset.id2label[int(l)] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
            "all_metrics": all_metrics
        }
    
    def model(self,push_to_hub=False,model_name="medical-bert"):
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            id2label=self.dataset.id2label,
            label2id=self.dataset.label2id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)

        if push_to_hub is True:
            access_token_write = config['huggingface']['ACCESS_TOKEN_WRITE']
            login(token = access_token_write)
            print("Login to huggingface succesful")

        epochs = int(config['bert']['EPOCHS'])
        learning_rate = float(config['bert']['LEARNING_RATE'])

        args = TrainingArguments(
            model_name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            weight_decay=0.01,
            push_to_hub=push_to_hub
        )

        self.prepare_data()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )

        return trainer
    
    def train(self,push_to_hub = None,model_name=None):
        trainer = self.model(push_to_hub=push_to_hub,model_name=model_name)
        trainer.train()
        trainer.push_to_hub()

           