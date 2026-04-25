import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

chat = client.chats.create(model="gemini-3-flash-preview")

pergunta = input("Digite algo para o chat bot: ")

resposta = chat.send_message(pergunta)

print(resposta.text)

print(chat.get_history())