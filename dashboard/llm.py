#from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv


class Llm_chain:
    def __init__(self):
        self._data_path = '../data/pdf_slides.txt'  
        self.conversation_chain = None
        self._setup_chain()

    # def _get_doc_text(self):
    #     with open(self._data_path, 'rb') as file:
    #         text = ''
    #         pdf_reader = PdfReader(file)
    #         for page_num in range(len(pdf_reader.pages)):
    #             page = pdf_reader.pages[page_num]
    #             text += page.extract_text()
    #     return text
        
    def _get_doc_text(self):
        with open(self._data_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def _get_text_chunks(self, text):
        chunk_size = 500
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=100,
            length_function=len
        )
        chunks = splitter.split_text(text)
        return chunks

    def _get_vectorstore(self, chunks):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore

    def _get_conversation_chain(self, vectorstore):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), vectorstore.as_retriever(), memory=memory)
        return conversation_chain

    def _setup_chain(self):
        #os.environ["OPENAI_API_KEY"] = "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"
        load_dotenv()
        text = self._get_doc_text()
        chunks = self._get_text_chunks(text)
        embeddings = self._get_vectorstore(chunks)
        self.conversation_chain = self._get_conversation_chain(embeddings)

    def generate_answer(self, user_input):
        if self.conversation_chain is None:
            raise ValueError("Conversation chain is not set. Call set_pdf_path() first.")
        response = self.conversation_chain({'question': user_input})
        return response['answer']


if __name__ == '__main__':
    llm_chain = Llm_chain()
    user_input = input('You:')
    try:
        answer = llm_chain.generate_answer(user_input)
        print('Bot:', answer)
        print('\n')
    except ValueError as e:
        print("Error:", e)


# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai.chat_models import ChatOpenAI
# # from langchain_community.llms import Ollama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import CharacterTextSplitter
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.document_loaders import TextLoader
# from langchain_core.runnables import RunnablePassthrough
# import pandas as pd
# import logging

# class Llm_chain:
#     def __init__(self):
#         self.loader = CSVLoader(file_path='../data/posts.csv')
#         self.documents = self.loader.load()
#         # self.docs = text_splitter.split_documents(documents)
#         self.embeddings = OpenAIEmbeddings()
#         self.db = FAISS.from_documents(self.documents, self.embeddings)
#         self.retriever = self.db.as_retriever()

#     def retrieve(self, query):
#         if not isinstance(query, str):
#             logging.error(f"Query must be a string, received: {type(query)}")
#             return []
#         similar_response = self.db.similarity_search(query, k=3)
#         page_contents_array = [doc.page_content for doc in similar_response]
#         print(page_contents_array)
#         return page_contents_array

    

#     def generate_response(self, question, classroom):
#         llm = ChatOpenAI(openai_api_key='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', model="gpt-4-turbo-preview")
#         template = """ 
#             You are a teaching assistant for {classroom}.
#             I will share a student's message with you and you will give me the best answer that
#             I should send to this student based on past best practices,
#             and you will follow ALL of the rules below:
            
#             1/ Response should be very similar or even identical to the past best practies,
#             in terms of length, tone of voice, logical arguments and other details.
#             2/ If the best practices are irrelevant, then try to mimic the style of the best practice to student's message, if name is anonymous then state them as student.

#             Here are some lecture embeddings as examples:
# S
#             Below is a message I received from the student:
#             {question}
#         """

#         prompt = ChatPromptTemplate.from_template(template)

#         def format_docs(self, docs):
#             return "\n\n".join([d.page_content for d in docs])

#         chain = (
#             prompt
#             | llm
#             | StrOutputParser()
#         )

#         response = chain.invoke(
#             {
#                 "classroom": classroom, 
#                 # "context": format_docs(self.retriever), 
#                 "question": question
#             }
#         )
#         return response
