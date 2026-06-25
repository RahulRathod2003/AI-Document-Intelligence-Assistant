from groq import Groq

client = Groq(api_key="GROQ_API_KEY")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain transformer in simple words"
        }
    ],
    model="llama-3.3-70b-versatile"
)

print(chat_completion.choices[0].message.content)