from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

class Llm_chain:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    template = """
    You are a {role} for {classroom}.
    I will share a prospect's message with you and you will give me the best answer that
    I should send to this prospect based on past best practies,
    and you will follow ALL of the rules below:
    1/ Response should be very similar or even identical to the past best practies,
    in terms of length, ton of voice, logical arguments and other details
    2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message
    Below is a message I received from the prospect:
    {message}
    Here is a list of best practies of how we normally respond to prospect in similar scenarios:
    {best_practice}
    Please write the best response that I should send to this prospect:
    """
    prompt = PromptTemplate(
        input_variables=["role","classroom", "message", "best_practice"],
        template=template
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    def create_vector_store(self, file):
        loader = CSVLoader(file_path=file)
        documents = loader.load()
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(documents, embeddings)

    def retrieve_info(self, query):
        similar_response = db.similarity_search(query, k=3)
        page_contents_array = [doc.page_content for doc in similar_response]
        return page_contents_array

    def generate_response(self, message):
        best_practice = retrieve_info(message)
        response = chain.run(message=message, best_practice=best_practice)
        return response

if __name__ == '__main__':
    load_dotenv()
    llm_chain = Llm_chain()