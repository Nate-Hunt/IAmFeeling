import os
import dotenv
import openai
from langchain.llms import openai
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from dotenv import load_dotenv

template = "Please evaluate the following text and categorize phrases by emotion, in bullet form: "

load_dotenv()

llm = ChatOpenAI()

loader = CSVLoader('data/emotion_words.csv')

index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(template + input("How are you feeling today? "), llm))