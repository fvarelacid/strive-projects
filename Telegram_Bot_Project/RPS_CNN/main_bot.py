import os
import telebot
import random
import requests
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from main import Net

import sys
sys.path.append("/Users/franciscovarelacid/Desktop/Strive/strive-projects/Telegram_Bot_Project/RPS_CNN/handgestures")
from handgestures.transform_image import transform_single_image

with open('static/config.json', 'r') as f:
 token = json.load(f)

bot = telebot.TeleBot(token["telegramToken"])
x = bot.get_me()
print(x)


choices = ['rock', 'paper', 'scissors']
computer_choice = random.choice(choices)


@bot.message_handler(commands=['play'])
def start(message):
   bot.send_message(message.chat.id, '''Upload an Image of YOUR HAND showing one of the gestures:
   -rock 
   -paper
   -scissors ''')

def user_input(message):
  request = message.text.lower()
  if request not in ['rock', 'paper', 'scissors']:
    return False
  else:
    return True

@bot.message_handler(content_types=['photo'])
def photo(message):
    
    fileID = message.photo[-1].file_id
    
    file_info = bot.get_file(fileID)
    
    downloaded_file = bot.download_file(file_info.file_path)

    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    #Transform the Image into a Skeleton
    img = transform_single_image("/Users/franciscovarelacid/Desktop/Strive/strive-projects/Telegram_Bot_Project/RPS_CNN/image.jpg")
    img = torch.Tensor(img)

    #Transform the Skeleton into a flattened array
    # img_arr = np.array(img, dtype = int)
    # img_arr = img_arr.flatten()
    # class_data_trail= pd.DataFrame(img_arr)
    # class_data_trail=class_data_trail.transpose()

    #import the model from RPS_net.pth
    save_path = 'RPS_net.pth'
    # model = Net()
    model = Net()
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    model.eval()

    # Generate prediction
    rps_class = model(img.reshape(1, 1, 128, 128))

    # Predicted class value using argmax
    predicted_class = np.argmax(rps_class.detach().numpy())

    if predicted_class == 0:
      bot.send_message(message.chat.id, "You Give ROCK!")
      player_choice = 'rock'
    elif predicted_class == 1:
      bot.send_message(message.chat.id, "You Give PAPER!")
      player_choice = 'paper'  
    else:
      bot.send_message(message.chat.id, "You Give SCISSORS!")
      player_choice = 'scissors'

    computer_choice = random.choice(choices)

    bot.send_message(message.chat.id,"PC picked: %s" % computer_choice)  

    if player_choice == computer_choice:
      bot.send_message(message.chat.id, "It's a Tie")      
    elif player_choice == 'rock' and computer_choice == 'scissors':
      bot.send_message(message.chat.id, "Player wins!")
    elif player_choice == 'scissors' and computer_choice == 'paper':
      bot.send_message(message.chat.id, "Player wins!")
    elif player_choice == 'paper' and computer_choice == 'rock':
      bot.send_message(message.chat.id, "Player wins!")
    else:
      bot.send_message(message.chat.id, "PC wins!")  







@bot.message_handler(func=user_input)
def send_output(message):
  player_choice = message.text.lower()

  computer_choice = random.choice(choices)

  if player_choice == computer_choice:
    bot.send_message(message.chat.id, "It's a Tie")      
  elif player_choice == 'rock' and computer_choice == 'scissors':
      bot.send_message(message.chat.id, "Player wins!")
  elif player_choice == 'scissors' and computer_choice == 'paper':
      bot.send_message(message.chat.id, "Player wins!")
  elif player_choice == 'paper' and computer_choice == 'rock':
      bot.send_message(message.chat.id, "Player wins!")
  else:
      bot.send_message(message.chat.id,"PC picked: %s" % computer_choice)  
      bot.send_message(message.chat.id, "PC wins!")
      
    
bot.polling()