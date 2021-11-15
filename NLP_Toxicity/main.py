##### Imports #####
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging
import numpy as np
import streamlit as st

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel(
    "bert",
    "content/checkpoints",
    use_cuda=False,
    from_tf=False,
    from_flax=False,
    num_labels=6
)


##### Comment Preprocessing & Structuring #####
def return_comment_labels(text):
    predictions, raw_outputs = model.predict([text])
    predictions = predictions[0]

    text_results = []

    for prediction in predictions:
      if prediction == 1:
        text_results.append('YES')
      else:
        text_results.append('NO')

    final_results = dict(zip(LABEL_COLUMNS, text_results))
                
    return final_results

##### Streamlit App #####

st.title("NLP Toxicity")
st.write("Lets check for your comment toxicity!")

user_input = st.text_input("Input text below")

if st.button("Predict Toxicity"):
    st.write(return_comment_labels(str(user_input)))



