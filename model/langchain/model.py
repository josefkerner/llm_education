from rec_service.model.model import Model
from typing import List,Dict
from rec_service.utils.secret.secret_manager import SecretManager
from langchain.llms import OpenAI
import tiktoken
import os
class LangChainWrapper(Model):
    def __init__(self, cfg: Dict):
        self.total_tokens_used = 0
        self.secret_manager = SecretManager()
        os.environ['OPENAI_API_KEY'] = self.secret_manager.get_open_ai_token()
        self.llm = OpenAI(temperature=0.0,
                          max_tokens=1200,
                          model_name="text-davinci-003")

    @staticmethod
    def estimate_tokens(prompt):
        enc = tiktoken.get_encoding("cl100k_base")
        encoded = enc.encode(prompt)
        return len(encoded)

    def generate_batch(self,prompts: List[str]):
        print(f"Generating batch of length {len(prompts)}")
        answers = []
        for prompt in prompts:
            print(prompt)
            #answer = llm(prompt)
            #answers.append(answer)
        return answers


    def generate(self, prompts: List[str]):
        '''
        Generate text from the given prompts
        :param prompts:
        :return:
        '''

        answers = ["Generated"]*len(prompts)
        print(f"Generating length {len(prompts)}")
        return answers
        '''
        p = mp.Pool(petr_strejc)
        func = partial(LangChainWrapper.generate_batch,self.llm)
        answers = p.map(func,prompts)
        
        for prompt in prompts:
            self.total_tokens_used += LangChainWrapper.estimate_tokens(prompt)
        answers = self.llm(prompts)
        '''
        return answers

