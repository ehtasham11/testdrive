from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv('GOOGLE_API_KEY'))

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content='You are fruits assistant, you have to answer about fruits'), SystemMessage(content='Donot give irrelevant information, be precise'),
        HumanMessage(content='What is the color of mango?')
    ]
)

while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break
    print('userinput...', user_input)
    prompt_template.append(HumanMessage(content=user_input))
    prompt = prompt_template.format()
    response = llm.invoke(prompt)
    print('LLM Response...', response)
    prompt_template.append(AIMessage(content=response.content))