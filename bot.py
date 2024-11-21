from google import generativeai as genai

API_KEY = "AIzaSyAnMtn5b8CE9OgGpMpW13zd8v4H1pnuBxA"

SYSTEM_PROMPT = open("system_prompt.txt").read()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": [SYSTEM_PROMPT]
    },
    {
        "role": "model",
        "parts": [
            "Understood. I will act as an AI mental health support assistant, following the guidelines you've "
            "provided. How may I assist you today?"]
    }
])


def respond_to(query: str) -> str:
    response = chat.send_message(query)
    return response.text
