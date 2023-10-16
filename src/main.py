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

exit = False
load_dotenv()
llm = ChatOpenAI()
# loader1 = CSVLoader('data/emotion_words.csv')
loader2 = CSVLoader('data/Emotion_final.csv')

index = VectorstoreIndexCreator().from_loaders([loader2])

while exit != True:
    template = "Please evaluate the following text and categorize phrases by emotion, in bullet form. Use this template: Emotion: Phrase. Use the categories happy, sad, surprised, fearful, angry, disgusted, and bad."
    print("__________________________________________")
    user_response = input("How are you feeling today? ")
    if user_response == "Exit" or user_response == "exit":
        exit = True
    elif user_response != 'Exit' or user_response != 'exit':
        print(index.query(template + user_response, llm) + "\n")

print("__________________________________________")