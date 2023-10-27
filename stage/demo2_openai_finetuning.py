'''
This file will test finetuning on a specific dataset
'''
from typing import List, Dict
from model.gpt3.gpt3_model import GPT3_model
class OpenAIFinetuning:
    def __init__(self):
        model_name = 'gpt3'
        self.llm = GPT3_model(
            cfg={
                'model_name': model_name,
            }
        )

    def test_finetuning(self):
        '''
        Will test finetuning on a specific dataset
        :return:
        '''
        cv = """
                John Doe is a Data Scientist with 3 years of experience.
                He has worked on multiple projects in the past.
                He has a Master's degree in Computer Science.
                He is proficient in Python, SQL, and Machine Learning.
                He knows how to use TensorFlow,PyTorch and Computer Vision libs.
                """
        role = """
                We are looking for a Data Scientist with 3 years of experience.
                The candidate should have a Master's degree in Computer Science.
                The candidate should be proficient in Python, SQL, and Machine Learning.
                The candidate should know how to use TensorFlow and PyTorch.
                """
        prompt = f"""
                Generate a recommendation if candidate a fit for the job role based on role requirements and candidate experience.
                If a candidate is not a fit for the job role, generate only string : "Not a fit".
                Candidate name: John Doe,
                Candidate experience: {cv}
                Job role requirements: {role}
                Recommendation:
                """

        self.llm.generate([prompt])

if __name__ == '__main__':
    OpenAIFinetuning = OpenAIFinetuning()
    OpenAIFinetuning.test_finetuning()

