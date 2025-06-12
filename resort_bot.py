from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error
from datetime import datetime

set_verbosity_error()

text_generation_pipeline = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", max_new_tokens=50)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

template = """
Today is {current_time}. You are Mad Bot, a helpful assistant for a resort owner, answering customer questions.
The customer asked: "{question}"
"""

prompt = PromptTemplate.from_template(template)
chain = prompt | llm

while True:
    dt = datetime.now()
    print("\n\n-------------------------------")
    user_question = input("How may I help you ? (q to quit)\n> ")
    print("\n\n")
    if user_question == "q":
        break
    for chunk in chain.stream({"current_time":dt.strftime("%d %B %Y"), "question":user_question}):
        print(chunk, end="", flush=True)
    print()
