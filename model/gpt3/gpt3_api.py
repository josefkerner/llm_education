import openai
from tenacity import (retry,stop_after_attempt,wait_random_exponential)
from rec_service.utils.secret.secret_manager import SecretManager

@retry(wait=wait_random_exponential(min=1,max=60), stop=stop_after_attempt(5))
def call_gpt3(prompt,config):
    '''
    Will test gpt3 based model with given prompt and config
    :param prompt:
    :return:
    '''

    openai.api_key = SecretManager.get_open_ai_token()
    max_tokens = config['max_tokens'] if 'max_tokens' in config else 256
    temperature = config['temperature'] if 'temperature' in config else 0.0
    model = config['model_name'] if 'model_name' in config else "text-davinci-003"

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    answers = []

    for choice in response.choices:
        text = choice.text
        answers.append(text)
    return answers


