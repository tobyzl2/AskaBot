import os
import torch
import numpy as np
import pandas as pd

from ast import literal_eval
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from aska_bot.src.keyword_matching.data_generator import build_keyword_dataset


class KeywordDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.questions = np.array(df["question"])
        self.keywords = np.array(df["keywords"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.questions[item], " ".join(literal_eval(self.keywords[item]))


def get_input(questions, keywords, tokenizer):
    # tokenize question
    question_tokens = tokenizer.batch_encode_plus(questions, padding=True)
    input_ids = question_tokens["input_ids"]
    attention_mask = question_tokens["attention_mask"]

    # get label
    special_tokens = tokenizer.all_special_ids
    keyword_ids = tokenizer.batch_encode_plus(keywords)["input_ids"]
    labels = []
    for i in range(len(input_ids)):
        labels.append([1 if id in keyword_ids[i] and id not in special_tokens else 0 for id in input_ids[i]])

    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)


def get_dataloaders(keyword_dataset_path, json_path=None, batch_size=64, test_size=0.1, random_seed=42):
    # get dataframe
    if os.path.exists(keyword_dataset_path):
        df = pd.read_csv(keyword_dataset_path)
    elif json_path:
        df = build_keyword_dataset(json_path)
        df.to_csv(keyword_dataset_path)
    else:
        assert False

    # train test split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_seed, shuffle=False)

    # instantiate datasets
    keyword_train_dataset = KeywordDataset(df_train)
    keyword_test_dataset = KeywordDataset(df_test)

    # instantiate dataloaders
    keyword_train_dataloader = DataLoader(keyword_train_dataset, batch_size=batch_size, shuffle=True)
    keyword_test_dataloader = DataLoader(keyword_test_dataset, batch_size=batch_size, shuffle=True)

    return keyword_train_dataloader, keyword_test_dataloader
