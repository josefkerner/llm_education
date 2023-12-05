import glob
import os.path

from agents.vectara_agent import VectaraAgent
from utils.connectors.vectara_doc_loader import DocumentLoader
class ChatBot:
    def __init__(self):
        self.agent = VectaraAgent()
        self.document_loader = DocumentLoader()
        self.qa_client = self.agent.get_qa_agent(corpus_id=os.getenv('VECTARA_CORPUS_ID'))

    def add_docs(self):
        '''
        Will add docs to vector store
        :return:
        '''
        docs_paths = glob.glob(os.path.join("data/txt_files", "*.txt"))
        for doc_path in docs_paths:
            doc = self.document_loader.load_doc(doc_path)
            self.document_loader.add_doc_to_vector_store(doc)

    def ask(self,question: str):
        '''
        Answers user question
        :param question:
        :return:
        '''
        response = self.qa_client({"query": question})
        return response

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.add_docs()
    question = "How many new crystal based materials did Gnome project find?"
    answer = chatbot.ask(question=question)
    print(answer)
