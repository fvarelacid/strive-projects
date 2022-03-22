import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import nltk
import streamlit as st
from streamlit_chat import message
import random

nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "trained.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Gloria"

prev_tag = None

##### Streamlit App #####

st.set_page_config(
    page_title="Emergency Chatbot",
    page_icon=":robot:"
)

st.title("\U0001f691 Emergency Chatbot \U0001f691")



st.write("You are talking to our bot nurse, {}".format(bot_name))

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'prev_tag' not in st.session_state:
    st.session_state['prev_tag'] = None

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()

prev_tag = st.session_state['prev_tag']

if user_input:

    st.session_state.past.append(user_input)

    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output_model = model(X)
    _, predicted = torch.max(output_model, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output_model, dim=1)
    prob = probs[0][predicted.item()]

    print("Tag: ", tag)
    print("Probability: ", prob)

    if prob.item() > 0.8:
        for intent in intents['intents']:
            if tag == intent['tag']:
                if prev_tag == "appointment" and tag == "affirmation":
                    output = f"{random.choice(intent['responses'])}"
                elif prev_tag == "affirmation" and tag == "affirmation":
                    output = f"{random.choice(intent['responses2'])}"
                elif prev_tag == "appointment" and tag == "negation":
                    output = f"{random.choice(intent['responses'])}"
                elif prev_tag == "affirmation" and tag == "negation":
                    output = f"{random.choice(intent['responses2'])}"
                else:
                    output = f"{random.choice(intent['responses'])}"
                st.session_state.generated.append(output)
                print("Prev tag: ", st.session_state.prev_tag)
                st.session_state.prev_tag = tag
                print("Confirmed tag: ", tag)
    else:
        output = "I do not understand. Try to be more specific."
        st.session_state.generated.append(output)


if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    