import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("MODEL_ID"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("API_KEY"),
    temperature=0.5,
    max_tokens=256,
)



