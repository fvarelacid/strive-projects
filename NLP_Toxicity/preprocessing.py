import pandas as pd
from helper import clean_text

df = pd.read_csv('NLP_Toxicity/data/train.csv')
df['comment_text'] = df['comment_text'].map(lambda txt : clean_text(txt, remove_stopwords=False))
df.to_csv('train.csv', index=False)

df = pd.read_csv('NLP_Toxicity/data/test.csv')
df['comment_text'] = df['comment_text'].map(lambda txt : clean_text(txt, remove_stopwords=False))
df.to_csv('test.csv', index=False)