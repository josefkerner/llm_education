from langchain import PromptTemplate
from model.gpt3.gptturbo import GPT_turbo_model
from prompts.inContextExampleManager import inContextExampleManager

class OpenAIPrompter:
    def __init__(self):
        self.llm = GPT_turbo_model(
            cfg={}
        )

    def test_base_prompt(self):
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
        cv_example = """
        Petr Novak is a Data Scientist with 3 years of experience.
        He has worked on multiple projects in the past.
        He has a Master's degree in Computer Science.
        He is proficient in Python, SQL, and Machine Learning.
        He knows how to use TensorFlow and PyTorch.
        """
        prompt = f"""
        Generate a recommendation if candidate a fit for the job role based on role requirements and candidate experience.
        If a candidate is not a fit for the job role, generate only string : "Not a fit".
        Candidate name: John Doe,
        Candidate experience: {cv}
        Job role requirements: {role}
        Recommendation:
        """
        prompt_with_example = f"""
        Generate a recommendation if candidate a fit for the job role based on role requirements and candidate experience.
        If a candidate is not a fit for the job role, generate only string : "Not a fit".
        Example:
        ------------
        Candidate name: Petr Novak,
        Candidate experience: {cv_example}
        Job role requirements: {role}
        Recommendation: Petr is a fit for the job role based on his knowledge.
        ------------
        Candidate name: John Doe,
        Candidate experience: {cv}
        Job role requirements: {role}
        Recommendation:
        """
        prompt = prompt.replace('\n', '').replace('\t', '').replace('  ', '')
        prompt_with_example = prompt_with_example.replace('\n', '').replace('\t', '').replace('  ', '')
        message = [{"role": "user", "content": prompt}]
        result = self.llm.generate([message])
        message = [{"role": "user", "content": prompt_with_example}]
        result_with_example = self.llm.generate([message])
        print(result)
        print(result_with_example)

    def test_advanced_prompt(self):
        role = """
                We are looking for a Data Scientist with 3 years of experience.
                The candidate should have a Master's degree in Computer Science.
                The candidate should be proficient in Python, SQL, and Machine Learning.
                The candidate should know how to use TensorFlow and PyTorch.
                """
        cv_example = """
                Petr Novak is a Data Scientist with 3 years of experience.
                He has worked on multiple projects in the past.
                He has a Master's degree in Computer Science.
                He is proficient in Python, SQL, and Machine Learning.
                He knows how to use TensorFlow and PyTorch.
                """

        prompt = inContextExampleManager().get_prompt(
            role_summary=role,
            cv_summary=cv_example,
            name="Petr",
            surname="Novak",
        )
        print(prompt)
        prompt = prompt.replace('\n', '').replace('\t', '').replace('  ', '')
        message = [{"role": "user", "content": prompt}]
        result = self.llm.generate([message])
        print("Advanced prompt result:")
        print(result)


if __name__ == '__main__':
    prompter = OpenAIPrompter()
    prompter.test_base_prompt()
    prompter.test_advanced_prompt()

