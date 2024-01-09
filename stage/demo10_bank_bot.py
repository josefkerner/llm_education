'''
Product Catalog Chatbot
This chatbot will help the user to find the product they are looking for using RAG and LLM models.
'''
import pandas as pd
from typing import List
from agents.vectara_agent import VectaraAgent
from utils.connectors.vectara_doc_loader import DocumentLoader
from langchain.schema.document import Document
from model.gpt.gpt_chat import GPT_turbo_model
import os

from pydantic import BaseModel
from typing import List, Dict

class ChatbotResponse(BaseModel):
    answer: Dict
    more_context_needed: bool
    chat_history: List[str]
class ProductCatalogChatbot:
    def __init__(self):
        self.agent = VectaraAgent()
        self.document_loader = DocumentLoader()
        self.qa_client = self.agent.get_qa_agent(corpus_id=os.getenv('VECTARA_CORPUS_ID'))
        self.gpt_chat = GPT_turbo_model(
            cfg={
                'model_name': 'gpt-3.5-turbo-16k',
                'max_output_tokens': 300
            }
        )

    def add_docs(self):
        df = pd.read_excel('data/product_catalog.xlsx')
        for row in df.itertuples():
            doc = Document(
                page_content=str(row.description),
                metadata={
                    'source': str(row.id),
                }
            )
            self.document_loader.add_doc_to_vector_store(doc)

    def check_context_needed(self, chat_history: List[str]):
        '''
        Checks if more context is needed
        :param chat_history:
        :return:
        '''
        chat_history_str = ','.join(chat_history)
        prompt = f"""
        Verify if the given chat history is specific enough to answer a question from bank knowledge base.
        chat history should contain a specific question to which user seeks answer.
        Answer with only 'yes' or 'no'.
        Chat history: {chat_history_str}
        Answer:
        """
        prompt_dict = [
            {"role": "system", "content": "You are expert in banking domain."},
            {"role": "user", "content": prompt}
        ]
        answer = self.gpt_chat.generate([prompt_dict])
        print("context checking --------------------")
        print(answer)
        if 'yes' in str(answer).lower():
            return False
        else:
            return True

    def generate_follow_up_question(self, chat_history: List[str]):
        '''
        Generates follow up question
        :param chat_history:
        :return:
        '''
        chat_history_str = ','.join(chat_history)
        print("chat history:")
        print(chat_history_str)
        prompt = f"""
        Based on given chat history, generate a follow up question which will ask user
        to describe specific product features he wants.
        Generate question in the same language as the user requests.
        Chat history: {chat_history_str}
        Follow up question:
        """
        sys_prompt = f"""
        You are expert in product catalog domain. Answer questions related to product catalog domain.
        Always respond in the same language as the user requests.
        Never use familiar language.
        """
        prompt_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        answer = self.gpt_chat.generate([prompt_messages])
        return answer[0]

    def chat(self, question: str, chat_history: List[str]):
        '''
        Answers user question
        :param question:
        :param chat_history:
        :return:
        '''
        chat_history.append(f"User: {question}")
        more_context_needed = self.check_context_needed(chat_history=chat_history)
        if more_context_needed:
            response = self.generate_follow_up_question(chat_history=chat_history)
            response = {
                'answer': response
            }
        else:
            response = self.recommend_product(chat_history=chat_history)

        response = ChatbotResponse(
            answer=response,
            more_context_needed=more_context_needed,
            chat_history=chat_history
        )
        return response

    def recommend_product(self, chat_history: List[str]):
        '''
        Answers user question
        :param question:
        :return:
        '''
        chat_history_str = ','.join(chat_history)
        prompt = f"""
        Extract specific product features from given chat history
        Always extract features in English
        Example: --------------------
        'User: Hele chtěl bych bundu', 'Bot: Jaké konkrétní vlastnosti byste chtěl, aby tato bunda měla?', 'User: noo asi dobrej vodní sloupec', 'Bot: Jaký by měl být vodní sloupec této bundy?', 'User: aspoň 50mm'
        Extracted product features: jacket, water column of at least 50mm
        -----------------------------
        Chat history: {chat_history_str}
        Extracted product features:
        """
        prompt_messages = [
            {"role": "system", "content": "You are expert in product catalog domain. Answer questions related to product catalog domain."},
            {"role": "user", "content": prompt}
        ]
        product_features = self.gpt_chat.generate([prompt_messages])
        product_features = product_features[0]
        print("product features:")
        print(product_features)

        prompt = f"""
        {product_features}
        """
        import json
        response = self.qa_client({"query": prompt})
        response = json.loads(response['result'])
        response = {
            'answer': response
        }

        return response

    def converse(self):
        '''
        Converse with the chatbot
        :return:
        '''
        chat_history = []
        while True:
            question = input('You: ')
            response = self.chat(question=question, chat_history=chat_history)
            print('Bot: ', response.answer)
            response.chat_history.append(f"Bot: {response.answer['answer']}")
            chat_history = response.chat_history

if __name__ == "__main__":
    chatbot = ProductCatalogChatbot()
    #chatbot.add_docs()
    print("Welcome to the product catalog chatbot. I will help you to find the product you are looking for.")
    #question = "Hele chtěl bundu kámo"
    #question = "I would like a jacket which I can use in mountains and has a water column of at least 50mm?"
    #answer = chatbot.chat(question=question, chat_history=[])
    chatbot.converse()


