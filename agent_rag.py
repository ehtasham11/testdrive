from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent, initialize_agent, create_structured_chat_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub

from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM (in our case it is GoogleGenerativeAI)
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv('GOOGLE_API_KEY'))

searchTool = TavilySearchResults( tavily_api_key=os.getenv('TAVILY_API_KEY'))

loader = WebBaseLoader("https://www.skillmatch.tech")
docs = loader.load()

documents = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(docs)

vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model='models/embedding-001'))

retreiver = vector.as_retriever()

retreiver_tool = create_retriever_tool(retreiver, "skillmatch_search", "Search for information about skillmatch website. For any questions related to skillmatch, You must use this tool")

tools = [searchTool, retreiver_tool]

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executor = initialize_agent(tools, agent,agent_type="tool-calling",  verbose=True)
# agent_executor = create_structured_chat_agent(tools, agent,agent_type="tool-calling",  verbose=True)

message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor, 
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


while True:
    agent_with_chat_history.invoke(
        {"input" : input("How I can help you today? : ")},
        config={"configurable": {"session_id": "test123"}}
    )
    