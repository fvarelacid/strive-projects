# Rock Paper Scissors Game with Telegram Bot

Here's me beating the bot:

![demo gif](./images/rps_bot.gif)

What's about: 
Telegram Chatbot that plays Rock Paper Scissors with you.

The user starts playing with the bot by sending an image of his own hand showing Rock, Paper or Scissors. The trained model will predict what
is that image representing and return if the user won, tied or lost.

The data was trained on a convolutional neural network model.

## How to run it with your bot:

1. Create a bot on telegram, follow this link: https://sendpulse.com/knowledge-base/chatbot/create-telegram-chatbot

2. Copy the token to the config.json file in the static folder

3. Install requirements
```
pip install -r requirements.txt
```

4. Activate bot with script below
```
python main_bot.py
```
5. If you got something like this it means your bot is running!
![Example Bot Running](./images/example_bot_run.gif)

6. Start bot or type '/start' and good luck!
![Good Luck](./images/good-luck-liam.gif.gif)


Any questions?

Just hit me up.

Francisco Varela Cid
Jan 2022