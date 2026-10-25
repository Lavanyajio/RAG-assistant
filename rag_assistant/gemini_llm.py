import google.generativeai as genai


class GeminiChatModel:

    def __init__(self, api_key, model="gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def chat(self, messages, **kwargs):

        prompt = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in messages
        )

        response = self.model.generate_content(prompt)

        return response.text