# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse
# import time

# # Set to store visited URLs to avoid duplication
# visited_urls = set()

# # Function to check if URL is internal (same domain)
# def is_internal_link(base_url, link):
#     # Parse base and link URLs
#     base_domain = urlparse(base_url).netloc
#     link_domain = urlparse(link).netloc

#     # If the link domain is empty or matches the base domain, it's internal
#     return link_domain == '' or base_domain == link_domain

# # Function to crawl a page and extract all links
# def crawl_page(url, base_url, file):
#     if url in visited_urls:
#         return  # If already visited, skip
    
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, 'html.parser')

#         # Write page content to file
#         file.write(f"\n--- Page: {url} ---\n")
#         file.write(soup.get_text())

#         # Mark the URL as visited
#         visited_urls.add(url)

#         # Find all links on the current page
#         for link in soup.find_all('a', href=True):
#             full_link = urljoin(base_url, link['href'])
            
#             # Check if the link is internal and hasn't been visited yet
#             if is_internal_link(base_url, full_link) and full_link not in visited_urls:
#                 time.sleep(1)  # Sleep for a second between requests to avoid overwhelming the server
#                 crawl_page(full_link, base_url, file)

#     except Exception as e:
#         print(f"Error crawling {url}: {e}")

# # Function to start the crawling process
# def crawl_website(base_url):
#     # Open file to save website content
#     with open('website_full_content.txt', 'w', encoding='utf-8') as file:
#         crawl_page(base_url, base_url, file)

# # Example usage:
# website_url = 'https://skillmatch.tech'  # Replace with the URL of the website
# crawl_website(website_url)


from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv("GOOGLE_API_KEY"))

# STEP: 1 Load the data
try:
    loader = TextLoader('website_full_content.txt')
except Exception as err:
    print('error occured', err)
    
# STEP: 2 Create Embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# STEP: 3 Split the Text into chunks
textSplitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# STEP: 4 Create the index with embedding and text splitter
index_creator =  VectorstoreIndexCreator(embedding=embedding,text_splitter=textSplitter)

index = index_creator.from_loaders([loader])

while True:
    human_message = input('Ask me any question>>> ')
    response = index.query(human_message, llm=llm)
    print(response)