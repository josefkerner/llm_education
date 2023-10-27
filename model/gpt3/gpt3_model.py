from model.model.model import Model
import openai
from tenacity import (retry,stop_after_attempt,wait_random_exponential)
from typing import List, Dict
import tiktoken
import os
if os.name == 'nt':
    os.environ['REQUESTS_CA_BUNDLE'] = "C:/python/openai/openai.crt"

    tiktoken_cache_dir = "C:/python/openai/rec_service/config/tiktoken_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

class GPT3_model(Model):
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.prompt_template = cfg['prompt_template'] if 'prompt_template' in cfg else None
        self.max_output_tokens = cfg['max_output_tokens'] if 'max_output_tokens' in cfg else 256
        self.MAX_INPUT_TOKENS= 4096 - self.max_output_tokens
        openai.api_key = os.environ["OPENAI_API_KEY"]



    def estimate_tokens(self,prompt):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        encoded = self.encoder.encode(prompt)
        return len(encoded)

    def split_chunks(self,prompts, max_tokens: 5000):
        """Yield successive n-sized chunks from lst."""
        chunks = []
        all_ids = []
        chunk = []
        chunk_ids = []
        chunk_tokens = 0
        MAX_CHUNK_ITEMS = 19
        for i, prompt in enumerate(prompts):
            prompt_token_len = self.estimate_tokens(prompt)

            if (chunk_tokens + prompt_token_len) > max_tokens or len(chunk) > MAX_CHUNK_ITEMS:
                chunks.append(chunk)
                all_ids.append(chunk_ids)
                chunk_tokens = prompt_token_len
                chunk = [prompt]  # append to next chunk
                chunk_ids = [i]

            else:
                chunk.append(prompt)
                chunk_ids.append([i])
                chunk_tokens = chunk_tokens + len(prompt)
        if chunk:
            all_ids.append(chunk_ids)
            chunks.append(chunk)

        return chunks, all_ids

    def generate(self, prompts: List[str],temp: float= 0.0):
        '''
        Will generate content
        :param text:
        :return:
        '''
        if self.prompt_template is not None:
            prompts =[ f"{self.prompt_template} {text}"
                       for text in prompts
                       ]

        all_answers = []

        chunks, all_ids = self.split_chunks(prompts, max_tokens=self.MAX_INPUT_TOKENS)
        all_chunk_ids = []
        for chunk, chunk_ids in zip(chunks, all_ids):
            assert len(chunk) != 0
            all_chunk_ids = all_chunk_ids + chunk_ids
            answers = self.call_gpt3(
                prompts=chunk,
                config=self.cfg)
            if all_answers is None:
                all_answers = answers
            else:
                all_answers = all_answers + answers
        assert len(prompts) == len(all_chunk_ids)
        return all_answers

    def generate_batch(self, prompts : List[str]):
        pass


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def call_gpt3(self,prompts:List[str], config: Dict):
        '''
        Will test gpt3 based model with given prompt and config
        :param prompt:
        :return:
        '''


        max_tokens = config['max_output_tokens'] if 'max_output_tokens' in config else 256
        temperature = config['temperature'] if 'temperature' in config else 0.0
        model = config['model_name'] if 'model_name' in config else "text-davinci-003"

        response = openai.Completion.create(
            engine=model,
            prompt=prompts,
            #model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        answers = []

        for choice in response.choices:
            text = choice.text
            answers.append(text)
        return answers
