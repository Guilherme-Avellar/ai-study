package br.com.guilherme;

import com.google.genai.Client;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.Chat;
import java.util.Scanner;

public class GenerateTextFromTextInput {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Client client = new Client();
        Chat chat = client.chats.create("gemini-3-flash-preview");

        System.out.print("Digite o prompt: ");
        String pergunta = sc.nextLine();

        GenerateContentResponse resposta = chat.sendMessage(pergunta);

        System.out.println(resposta.text());
    }
}
