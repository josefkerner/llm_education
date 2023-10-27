from model.model import Model
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

'''
This class is currently not used, since llama is not deployed in WG environment
'''


class LlamaModel(Model):
    def __init__(self, cfg: Dict):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'],
                                                        trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # Fix for fp16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model_bf16 = AutoModelForCausalLM.from_pretrained(
            cfg['model_name'],
            quantization_config=quantization_config
        )
        self.model_bf16.config.use_cache = False
        self.model_bf16.config.pretraining_tp = 1

    def generate(self, prompts: List[str], temp:float = 0.0):
        '''
        Will generate a response for each prompt
        :param prompts:
        :param temp:
        :return:
        '''
        prompts = [f"Q: {prompt}\nA:" for prompt in prompts]
        device = "cuda:0"
        inputs = self.tokenizer(prompts, return_tensors="pt").to(device)
        outputs = self.model_bf16.generate(**inputs,
                                           max_new_tokens=35
                                           )
        answers = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return answers