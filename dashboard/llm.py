from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
# from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import logging

class Llm_chain:
    def __init__(self):
        self.loader = CSVLoader(file_path='../data/posts.csv')
        self.documents = self.loader.load()
        
        self.embeddings = OpenAIEmbeddings()
        self.faiss_db = FAISS.from_documents(self.documents, self.embeddings)

    def retrieve_info(self, query):
        if not isinstance(query, str):
            logging.error(f"Query must be a string, received: {type(query)}")
            return []
        similar_response = self.faiss_db.similarity_search(query, k=3)
        page_contents_array = [doc.page_content for doc in similar_response]
        print(page_contents_array)
        return page_contents_array

    def generate_response(self, message, classroom):
        existing_posts = self.retrieve_info(message)
        existing_posts_str = "\n".join(existing_posts) if isinstance(existing_posts, list) else existing_posts
        print(existing_posts_str)

        llm = ChatOpenAI(openai_api_key='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', model="gpt-4-turbo-preview")
        # llm = Ollama(model="llama2")
        prompt = ChatPromptTemplate.from_template("""You are a teaching assistant for {classroom}.
        I will share a student's message with you and you will give me the best answer that
        I should send to this student based on past best practices,
        and you will follow ALL of the rules below:
        
        1/ Response should be very similar or even identical to the past best practies,
        in terms of length, tone of voice, logical arguments and other details.
        2/ If the best practices are irrelevant, then try to mimic the style of the best practice to student's message, if name is anonymous then state them as student.

        Below is a message I received from the student:
        {message}
        """
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        response = chain.invoke(
            {
            "classroom":classroom, 
            "message": message, 
            # "existing_posts": existing_posts_str
            }
        )
        return response