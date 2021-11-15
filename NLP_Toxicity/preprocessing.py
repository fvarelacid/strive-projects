from helper import clean_text
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from torchtext.vocab import FastText
from dataset import encoder


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


##### Cleaning Floats & Outlayers #####
fasttext = FastText("simple")

def clean_floats_long(dataf):
    list_index = []
    for i in range(len(dataf)):
        if (type(dataf["comment_text"][i]) == float) or (len(encoder(dataf["comment_text"][i], fasttext)) > 1000):
            list_index.append(i)
    dataf.drop(list_index, inplace=True)
    dataf.reset_index(inplace=True, drop=True)
    return dataf


def text_preprocessing(df):
    df = clean_floats_long(df)
    df = text_clean(df)
    df = clean_non_en(df)
    df.reset_index(inplace=True, drop=True)
    return df