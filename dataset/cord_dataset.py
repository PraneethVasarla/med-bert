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



class CORD_Dataset:
    def __init__(self,dataset_path=None,all_rows=False) -> None:
        if dataset_path is not None:
            with open(dataset_path,"r") as f:
                df = pd.read_csv(dataset_path)
            df.drop(columns=['Unnamed: 0'],inplace=True)
            df.rename(columns={"0":"Word","1":"sentence","2":"Sentence #"},inplace=True)
            
            if all_rows:
                self.df = df
            else:
                mini_set = df[df['Sentence #']<=25498]
                self.df = mini_set

            self.label_names = list(set([x for x in list(df['sentence'])]))
            self.df = self.df.groupby("Sentence #")[["Word","sentence"]].agg(list).reset_index()

            self.label2id = None
        else:
            raise Exception("Pass dataset path")
        
    def map_values(self,row):
        list_col_values = row['sentence']

        mapped_list = []

        for label in list_col_values:
            mapped_list.append(self.label2id[label])

        return mapped_list
        
    def prepare_hf_dataset(self):
        df = self.df.copy()


        self.label2id = {value:key for key,value in enumerate(self.label_names)}
        self.id2label = {key:value for key,value in enumerate(self.label_names)}

        df['bio_tags'] = df.apply(self.map_values, axis=1)

        self.df = df

        # Split the data into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

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