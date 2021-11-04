from sklearn.datasets import make_multilabel_classification
from torch.utils.data import Dataset
import torch
import numpy as np
from torchtext.vocab import FastText
from collections import Counter
from preprocessing import padding, encoder
from helper import preprocessing


class TrainData(Dataset):
    def __init__(self, df, max_seq_len=150): # df is the input df, max_seq_len is the max lenght allowed to a sentence before cutting or padding
        self.max_seq_len = max_seq_len
        
        counter = Counter()
        train_iter = iter(df.comment_text.values)
        self.vec = FastText("simple")
        self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0]) # replacing the vector associated with 1 (padded value) to become a vector of -1.
        self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0]) # replacing the vector associated with 0 (unknown) to become zeros
        self.vectorizer = lambda x: self.vec.vectors[x]
        self.label_toxic = df.toxic
        self.label_stoxic = df.severe_toxic
        self.label_obscene = df.obscene
        self.label_threat = df.threat
        self.label_insult = df.insult
        self.label_ihate = df.identity_hate
        sequences = [padding(encoder(preprocessing(sequence), self.vec), max_seq_len) for sequence in df.comment_text.tolist()]
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.label_toxic[i], self.label_stoxic[i], self.label_obscene[i], self.label_threat[i], self.label_insult[i], self.label_ihate[i]


def structure_dataset(dataset):
    com_array = []
    labs_array = []
    for i in range(len(dataset)):
        com_array.append(dataset[i][0])
        lab_array = []
        for j in range(1,7):
            lab_array.append(dataset[i][j])
        labs_array.append(lab_array)
    return np.array(com_array), np.array(labs_array)



class BinaryDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        features = self.x[index, :]
        labels = self.y[index, :]
        
        # we have 12 feature columns 
        features = torch.tensor(features, dtype=torch.float32)
        # there are 5 classes and each class can have a binary value ...
        # ... either 0 or 1
        label1 = torch.tensor(labels[0], dtype=torch.float32)
        label2 = torch.tensor(labels[1], dtype=torch.float32)
        label3 = torch.tensor(labels[2], dtype=torch.float32)
        label4 = torch.tensor(labels[3], dtype=torch.float32)
        label5 = torch.tensor(labels[4], dtype=torch.float32)
        label6 = torch.tensor(labels[5], dtype=torch.float32)

        return {
            'features': features,
            'toxic': label1,
            'severe_toxic': label2,
            'obscene': label3,
            'threat': label4,
            'insult': label5,
            'identity_hate': label6,
        }