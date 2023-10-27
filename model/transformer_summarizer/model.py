from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from model.model import Model
from typing import Dict
'''
This class implements HuggingFace Transformer model
'''
class TransformerModel(Model):
    def __init__(self, cfg: Dict):
        assert "model_name" in cfg, "model_name not in model config"
        model_name = cfg['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_output_tokens: int = 1000):
        '''
        Generates text
        :param prompt:
        :param max_output_tokens:
        :return:
        '''

        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(inputs,
                  max_new_tokens=max_output_tokens,
                do_sample=False)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text