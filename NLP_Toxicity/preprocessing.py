import pandas as pd
from helper import clean_text
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, tqdm_notebook
from torchtext.vocab import FastText


##### Cleaning Text #####

def text_clean(df):
    df['comment_text'] = df['comment_text'].map(lambda txt : clean_text(txt, remove_stopwords=False))
    return df

##### Droping non Engilsh Rows #####

def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)


def clean_non_en(dataf):
    list_index = []
    for i in range(len(dataf)):
        if (nlp(dataf["comment_text"][i])._.language['language'] != 'en'):
            list_index.append(i)
    dataf.drop(list_index, inplace=True)
    return dataf

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


##### Cleaning Floats & Outlayers #####
fasttext = FastText("simple")

def clean_floats_long(dataf):
    list_index = []
    for i in range(len(dataf)):
        if (type(dataf["comment_text"][i]) == float) or (len(encoder(dataf["comment_text"][i], fasttext)) > 1000):
            list_index.append(i)
    dataf.drop(list_index, inplace=True)
    dataf.reset_index(inplace=True)
    dataf.drop(columns=['index'], inplace=True)
    return dataf