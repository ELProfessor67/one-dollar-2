from openai import OpenAI
import dotenv
dotenv.load_dotenv()
client = OpenAI()
from typing import List,Dict

class LLM():
    model: str
    instruction: str
    chat_ctx: List[Dict[str, str]] = []
    def __init__ (self,instruction:str,model:str="gpt-4.1"):
        self.instruction = instruction
        self.model = model
        self.chat_ctx.append({
            "role": "system",
            "content": instruction,
        })


    def generate(self,text):
        self.chat_ctx.append({
            "role": "user",
            "content": text,
        })

        stream = client.responses.create(
            model=self.model,
            input=self.chat_ctx,
            stream=True,
        )

        for event in stream:
            if event.type == "response.output_text.done":
                self.chat_ctx.append({
                    "role": "user",
                    "content": event.text,
                })

                return event.text