import os,sys

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import DataCollatorForTokenClassification,AutoTokenizer,AutoModelForTokenClassification,get_scheduler
from accelerate import Accelerator
from huggingface_hub import Repository, get_full_repo_name,login
from tqdm.auto import tqdm
import evaluate
import configparser
config = configparser.ConfigParser()
config.read("config/project_config.ini")

class CutsomBertModel:
    def __init__(self,dataset=None,model_checkpoint="bert-base-uncased") -> None:
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
        batch_size = int(config['bert']['BATCH_SIZE'])
        self.tokenized_datasets = self.dataset.hf_dataset.map(
            self.tokenize_and_align_lables,batched=True,remove_columns=self.dataset.hf_dataset['train'].column_names
            )
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.label_names = list(self.dataset.label2id.keys())

        self.train_dataloader = DataLoader(
            self.tokenized_datasets["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=batch_size)
        
        self.eval_dataloader = DataLoader(
            self.tokenized_datasets["validation"], collate_fn=self.data_collator, batch_size=batch_size)
        
    def postprocess(self,predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        self.metric = evaluate.load("seqeval")
        return true_labels, true_predictions

    def model(self,push_to_hub=False,model_name="medical-bert"):
        
        learning_rate = float(config['bert']['LEARNING_RATE'])

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            id2label=self.dataset.id2label,
            label2id=self.dataset.label2id,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint,add_prefix_space = True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)

        weight_decay = float(config['bert']['WEIGHT_DECAY'])

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.accelerator = Accelerator()
        self.prepare_data()

        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            model, optimizer, self.train_dataloader, self.eval_dataloader
            )
        
        self.lr_scheduler = self.get_scheduler()

        if push_to_hub is True:
            self.login_huggingface()
        
    def get_scheduler(self):
        epochs = int(config['bert']['EPOCHS'])
        self.num_train_epochs = epochs
        num_update_steps_per_epoch = len(self.train_dataloader)
        self.num_training_steps = self.num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

        return lr_scheduler
    
    def login_huggingface(self,model_name = "finetuned-accelerate-ner"):
        access_token_write = config['huggingface']['ACCESS_TOKEN_WRITE']
        login(token = access_token_write)
        print("Login to huggingface succesful")
        model_name = model_name
        repo_name = get_full_repo_name(model_name)
        self.repo = Repository(self.output_dir, clone_from=repo_name)

    def train(self,push_to_hub = None,model_name=None):
        print("Change config.ini to pass the hyperparameters")
        
        self.model(push_to_hub=push_to_hub,model_name=model_name)
        self.output_dir = model_name
        progress_bar = tqdm(range(self.num_training_steps))
        for epoch in range(self.num_train_epochs):
            # Training
            self.model.train()
            for batch in self.train_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            self.model.eval()
            for batch in self.eval_dataloader:
                with torch.no_grad():
                    outputs = self.model(**batch)

                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                predictions = self.accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = self.accelerator.gather(predictions)
                labels_gathered = self.accelerator.gather(labels)

                true_predictions, true_labels = self.postprocess(predictions_gathered, labels_gathered)
                self.metric.add_batch(predictions=true_predictions, references=true_labels)

            results = self.metric.compute()
            print(
                f"epoch {epoch}:",
                {
                    key: results[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]
                },
            )

            # Save and upload
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(self.output_dir, save_function=self.accelerator.save)
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.output_dir)
                if push_to_hub is True:
                    self.repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False
                    )

    