import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchtext.vocab import FastText
from collections import Counter
from transformers import BertTokenizerFast as BertTokenizer
import pytorch_lightning as pl


LABEL_COLUMNS = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']


##### Token Encoder & Padding #####

def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0

def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]

def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + (max_seq_len - len(list_of_indexes))*[padding_index]
    return output[:max_seq_len]

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
        sequences = [padding(encoder(sequence, self.vec), max_seq_len) for sequence in df.comment_text.tolist()]
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


class ToxicCommentsDataset(Dataset):

  def __init__(
    self, 
    data: pd.DataFrame, 
    tokenizer: BertTokenizer, 
    max_token_len: int = 128
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    comment_text = data_row.comment_text
    labels = data_row[LABEL_COLUMNS]

    encoding = self.tokenizer.encode_plus(
      comment_text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      comment_text=comment_text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )


class ToxicCommentDataModule(pl.LightningDataModule):

  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=300):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def setup(self, stage=None):
    self.train_dataset = ToxicCommentsDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )

    self.test_dataset = ToxicCommentsDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2
    )

  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )