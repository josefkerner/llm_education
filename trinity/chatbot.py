import glob
import os.path
from typing import List
from agents.vectara_agent import VectaraAgent
from utils.connectors.vectara_doc_loader import DocumentLoader
from trinity.prompt_firewall import PromptFirewall
class ChatBot:
    PROMPT_FIREWALL_ENABLED = True
    def __init__(self):

        self.prompt_firewall = PromptFirewall()
        self.agent = VectaraAgent()
        self.document_loader = DocumentLoader()
        self.qa_client = self.agent.get_qa_agent(corpus_id=os.getenv('VECTARA_CORPUS_ID'))

    def add_docs(self):
        '''
        Will add docs to vector store
        :return:
        '''
        docs_paths = glob.glob(os.path.join("data/trinity", "*.txt"))
        for doc_path in docs_paths:
            doc = self.document_loader.load_doc(doc_path)
            self.document_loader.add_doc_to_vector_store(doc)

    def ask(self,question: str):
        '''        Answers user question
        :param question:
        :return:
        '''
        response = self.qa_client({"query": question})
        return response

    def chat(self, question: str, chat_history: List[str]):
        '''
        Chats with user
        :param question:
        :param chat_history:
        :return:
        '''
        if self.PROMPT_FIREWALL_ENABLED:
            if not self.prompt_firewall.verify_question(question=question):
                return "I'm sorry, I am not allowed to answer your question", chat_history
        response = self.qa_client({"query": question})
        chat_history.append(question)
        return response, chat_history

    def converse(self):
        '''
        Converse with user
        :return:
        '''
        chat_history = []
        while True:
            question = input("User: ")
            if question == 'exit':
                break
            response, chat_history = self.chat(question=question, chat_history=chat_history)
            print("Bot: ", response['result'])
            print("Chat history: ", chat_history)
            print("----------------------------------------------------------------------------------")

if __name__ == "__main__":
    chatbot = ChatBot()
    #chatbot.add_docs()
    question = ""
    question = "Jaká je základní sazba na účtu bonus při dobé vázanosti 24 měsíců pokud vložím více jak 12 milionů CZK?"
    #question = "Když si u vás otevřu účet, dostanu kartu?"
    #question = "Kolik stojí vedení spořícího účtu?"
    #question = "Jaké máte úrokové sazby na spořícím účtu když chci uložit milion korun?"
    #question = "Co potřebuji k založení účtu? A jde založit online?"
    #answer = chatbot.ask(question=question)
    #print(answer)
    chatbot.converse()
