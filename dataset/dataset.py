import os,sys
import json
import re

sys.path.append(os.getcwd())

from datasets import Dataset,DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import random
from nltk.corpus import wordnet



class NERDataset:
    def __init__(self,dataset_path=None) -> None:
        if dataset_path is not None:
            with open(dataset_path,"r") as f:
                self.df = json.load(f)

            self.label_names = None
            self.label2id = None
        else:
            raise Exception("Pass dataset path")
        
    def map_values(self,row):
        list_col_values = row['bio_tags']

        mapped_list = []

        for label in list_col_values:
            mapped_list.append(self.label2id[label])

        return mapped_list
    
    def split_sentence(self,sentence):

        words = re.split(r'((?<=\[)[^\]]+(?=\]))|[,.;:\s0-9]+', sentence)

        # Check for 'None' values and print warning messages
        for word in words:
            if word is None:
                pass

        # Remove 'None' values from the list
        words = [word for word in words if word is not None]

        # Apply rstrip() to non-None values
        words = [word.rstrip() for word in words]

        return words

    def remove_pattern(self,text):
        # Regular expression to match any number within brackets
        pattern_regex = re.compile('\[\d+\]')

        # Replace the pattern with an empty string
        text_without_pattern = pattern_regex.sub('', text)

        return text_without_pattern

    def remove_punctuation(self,text):
        # Regular expression to match punctuation characters
        punctuation_regex = re.compile('[\\.,!?\'\"]')

        # Replace punctuation characters with an empty string
        text_without_punctuation = punctuation_regex.sub('', text)

        return text_without_punctuation
        
    def create_bio_tags(self,row):
        BIO = []
        sentence = row['sentence']
        labels = row['labels']
        for word in sentence:
            if word in labels:
                BIO.append(labels[word])
            else:
                BIO.append("O")

        return BIO
    
    def synonym_replacement(self, sentence):
        synonyms = []
        for word in sentence:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())

        # If synonyms are found, replace a word with a synonym randomly
        if synonyms:
            random_synonym = random.choice(synonyms)
            return random_synonym
        else:
            return word

    def augment_data(self):
        augmented_sentences = []
        for index, row in self.df.iterrows():
            sentence = row['sentence']
            augmented_sentence = [self.synonym_replacement(word) for word in sentence]
            augmented_sentences.append(augmented_sentence)

        return augmented_sentences
        
    def prepare_hf_dataset(self):
        df = self.df.copy()

        tags = []
        for annots in self.df['examples']:
            annotations = [annotation['tag_name'] for annotation in annots['annotations']]
            tags.extend(annotations)

        tags = list(set(tags))

        bio_tags = []
        bio_tags.append("O")
        for tag in tags:
            bio_tags.append(f"B-{tag}")
            bio_tags.append(f"I-{tag}")

        self.label_names = bio_tags

        del tags

        df_new = pd.DataFrame(columns=['id','content','labels'])
        for annots in self.df['examples']:
            # print(f"{annots['id']}: {annots['content']}")
            labels = {annotation['value']:annotation['tag_name'] for annotation in annots['annotations']}
            # print("\n")
            # print(labels)
            new_labels = {}
            for label in labels:
                if len(label.split()) > 1:
                    # i = 0
                    for i,word in enumerate(label.split()):
                        if i == 0:
                            new_labels[word] = f"B-{labels[label]}"
                        else:
                            new_labels[word] = f"I-{labels[label]}"
                else:
                    new_labels[label] = f"B-{labels[label]}"

            new_row = {'id': annots['id'], 'content': annots['content'], 'labels':new_labels}

            # Append the new row to the dataframe
            df_new.loc[len(df_new)] = new_row
        
        df_new['content'] = df_new['content'].apply(self.remove_pattern)
        df_new['content'] = df_new['content'].apply(self.remove_punctuation)
        df_new['sentence'] = df_new['content'].apply(self.split_sentence)

        df_new['bio_tags'] = df_new.apply(self.create_bio_tags,axis=1)


        self.label2id = {value:key for key,value in enumerate(self.label_names)}
        self.id2label = {key:value for key,value in enumerate(self.label_names)}

        df_new['bio_tags'] = df_new.apply(self.map_values, axis=1)
        df_new.drop(['id','content','labels'],axis=1)

        self.df = df_new

        augmented_sentences = self.augment_data()

        df_new['augmented_sentence'] = augmented_sentences

        # Split the data into train and test sets
        train_df, test_df = train_test_split(df_new, test_size=0.25, random_state=42)

        # Split the train set into train and validation sets
        train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

        train_dataset = Dataset.from_pandas(train_df,preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df,preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df,preserve_index=False)

        dataset = DatasetDict()
        dataset['train'] = train_dataset
        dataset['validation'] = val_dataset
        dataset['test'] = test_dataset

        self.hf_dataset = dataset