import pandas as pd
from dataset import ToxicCommentDataModule
from model import BinaryModel
from preprocessing import text_preprocessing
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast as BertTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


BERT_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
MAX_TOKEN_COUNT = 512

# initialize the model
model = BinaryModel()
print('Model Uploaded')

##### Df Import #####
train_df = pd.read_csv('./data/train.csv')
print('Train_df created')

##### Label Columns #####
LABEL_COLUMNS = train_df.columns.tolist()[1:]

##### Splitting Train & Val #####
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

##### Balance the train df with toxic and non toxic #####
train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
train_non_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]
train_df = pd.concat([train_toxic, train_non_toxic.sample(15000)])
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

##### Text Preprocessing #####
train_df = text_preprocessing(train_df)
print('Train Df Preprocessed')
val_df = text_preprocessing(val_df)
print('Val Df Preprocessed')


# hyperparameters
N_EPOCHS = 5
BATCH_SIZE = 16


##### Data Module Creation #####
data_module = ToxicCommentDataModule(
  train_df,
  val_df,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_TOKEN_COUNT
)

##### Checkpoint Callback #####
checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)
logger = TensorBoardLogger("lightning_logs", name="toxic-comments")
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

trainer = pl.Trainer(
  logger=logger,
  checkpoint_callback=checkpoint_callback,
  callbacks=[early_stopping_callback],
  max_epochs=N_EPOCHS,
  gpus=1,
  progress_bar_refresh_rate=30
)

##### Training #####
trainer.fit(model, data_module)


#### Test the model #####
trainer.test()