from Client import client

chat = client.chats.create(model="gemini-3-flash-preview")

pergunta = input("Digite algo para o chat bot: ")

resposta = chat.send_message(pergunta)

print(resposta.text)