import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

chat = client.chats.create(model="gemini-3-flash-preview")

prompt = input("Digite o prompt para a LLM: ")
while prompt != "/exit":
    resposta = chat.send_message(prompt)
    print(resposta.text)
    prompt = input("Digite o prompt para a LLM: ")

print(chat.get_history())