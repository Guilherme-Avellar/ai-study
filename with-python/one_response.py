from Client import client

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explique como a IA funciona em poucas palavras",
)

print(response.text)