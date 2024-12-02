from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv('GOOGLE_API_KEY'))

loader = WebBaseLoader("https://www.skillmatch.tech")
docs = loader.load()

documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model='models/embedding-001'))

retreiver = vector.as_retriever()
# create a tool for skillmatch website queries
retreiver_tool = create_retriever_tool(retreiver, "skillmatch_search", "Search for information about skillmatch website. For any questions related to skillmatch, You must use this tool")