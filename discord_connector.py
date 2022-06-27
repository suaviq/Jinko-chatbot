import os
import json
import requests
import discord
from dotenv import load_dotenv
import dotenv
import tensorflow as tf
from tensorflow import keras
from training import *
from chatbot import *

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
client = discord.Client()


class DiscordConnector(discord.Client):
    def __init__(self):
        super().__init__()
        train_chatbot, train_data, batch_loss, input_text_processor, output_text_processor = get_model()
        train_chatbot.load_weights('model/chatbot-trained').expect_partial()
        chatbot = Chatbot(
            encoder=train_chatbot.encoder,
            decoder=train_chatbot.decoder,
            input_text_processor=input_text_processor,
            output_text_processor=output_text_processor)
        self.chatbot = chatbot

    @client.event
    async def on_ready(self):
        print(f'Logged in as {self.user}')
    
    @client.event
    async def on_message(self, message):
        if message.author.id == self.user.id:
            return

        if message.content.startswith('Jinko!'):
            await message.channel.send('Baka mi tai yare yare daze -,-')

        if message.content != 'Jinko!':
            input_text = str(message.content)
            result = self.chatbot.chatbot_unrolled([input_text])
            response = result['text'].numpy()[0].decode()
            print("incoming message: ", input_text, 'chatbot response: ', response)
            await message.channel.send(response)

def main():
    client = DiscordConnector()
    client.run(TOKEN)

if __name__ == '__main__':
    main()
