from rec_service.model.gpt3.gpt3_api import call_gpt3
from rec_service.model.gpt3.gptturbo import GPT_turbo_model
from time import sleep
from typing import Dict
import tiktoken
import os
#TODO
os.environ['REQUESTS_CA_BUNDLE'] = "C:/python/openai/openai.crt"
tiktoken_cache_dir = "C:/python/openai/rec_service/config/tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

class GPT3utils:
    def __init__(self):
        pass

    @staticmethod
    def estimate_tokens(prompt):
        enc = tiktoken.get_encoding("cl100k_base")
        encoded = enc.encode(prompt)
        return len(encoded)

    @staticmethod
    def process_gpt3(prompts, config: Dict):

        all_answers = None
        MAX_CHARS = config['max_chunk_chars'] if 'max_chunk_chars' in config else 4096
        chunks, all_ids = GPT_turbo_model.split_chunks(prompts, max_tokens=MAX_CHARS)

        all_chunk_ids = []
        for chunk, chunk_ids in zip(chunks, all_ids):
            all_chunk_ids = all_chunk_ids + chunk_ids
            if 'turbo' in config['model_type']:
                model = GPT_turbo_model(cfg=config)
                answers = model.generate(chunk, config=config)
            else:
                answers = call_gpt3(chunk, config=config)

            sleep(4)
            if all_answers is None:
                all_answers = answers
            else:
                all_answers = all_answers + answers
        assert len(prompts) == len(all_chunk_ids)
        return all_answers

