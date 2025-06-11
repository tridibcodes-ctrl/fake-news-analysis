from google import genai

client = genai.Client(api_key="AIzaSyC5YwiSnkg8gj1DNPLVv7gZCrsv7vIk4V0")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)