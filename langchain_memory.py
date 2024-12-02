from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv('GOOGLE_API_KEY'))

memory = ConversationBufferWindowMemory(k=5)

chain = ConversationChain(llm=llm, memory=memory)

while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break
    response = chain.invoke(user_input)
    print('Final Response', response)