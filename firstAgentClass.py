from langchain_google_genai import  GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableSequence

from dotenv import load_dotenv
import os

load_dotenv()

llm = GoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv('GOOGLE_API_KEY'))

template = PromptTemplate(template="You are a tool caller agent. The user asked to add two numbers. the input i need in the tool must be two numbers seperated by a comma {input}", input_variables=['input'])

@tool
def add_two_numbers(input_data: str) -> str:
    """ addition of two numbers. """
    print('input???', input_data)
    numbers = input_data.split(',')

    print('numbers???', numbers)
  # Check if at least two numbers are present
    if len(numbers) < 2:
        return "Please provide at least two numbers to add."

  # Try converting extracted values to integers (handle potential errors)
    try:
        num1 = int(numbers[0])
        num2 = int(numbers[1])
    except ValueError:
        return "Invalid input. Please provide only numbers."

    result = num1 + num2
    return f"The Sum of {num1} and {num2} is {result}"
    # print("input_datas", input_data)

    # try:
    #     numbers = input_data.split(',')
    # except Exception as e: 
    #     return input_data
    
    # print("numbers", numbers)
    # num1, num2 = int(numbers[0]), int(numbers[1])
    # result = num1 + num2
    # return f"The Sum of {num1} and {num2} is {result}"

chain = RunnableSequence(template, llm, add_two_numbers)

output = chain.invoke("I have 3 apples. now i have 1. how many apples in total")
print('output is', output)