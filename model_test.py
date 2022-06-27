from training import *
from chatbot import *

train_chatbot.load_weights('model/chatbot-trained')
chatbot = Chatbot(
    encoder=train_chatbot.encoder,
    decoder=train_chatbot.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor)

chatbot.chatbot_unrolled(['TEST'])
