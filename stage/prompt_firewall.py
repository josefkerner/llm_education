from typing import List, Dict
from model.gpt.gpt_chat import GPT_turbo_model
from prompts.inContextExampleManager_firewall import inContextExampleManager
class PromptFirewall:
    def __init__(self):
        self.gpt_chat = GPT_turbo_model(
            cfg={
                'model_name': 'gpt-3.5-turbo-16k',
                'max_output_tokens': 300
            }
        )
        self.example_manager = inContextExampleManager()
    def verify_question(self, question: str) -> bool:
        sys_prompt = f"""
        You are a helful assistant focused solely on banking domain.
        """
        user_prompt = self.example_manager.get_prompt(
            question=question,
        )
        prompt_dict = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        answer = self.gpt_chat.generate([prompt_dict])
        print(answer)
        if 'yes' in str(answer[0]).lower():
            return True
        else:
            return False


if __name__ == '__main__':
    firewall = PromptFirewall()
    firewall.verify_question('Can I buy american ETF called VOO from your bank given that I am in Czech Republic?')

