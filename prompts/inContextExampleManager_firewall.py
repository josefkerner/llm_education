from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector, SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from typing import List,Dict
import os, re, pandas
class inContextExampleManager:
    def __init__(self):
        datapath= "data/firewall.csv"
        self.examples = self.load_from_file(datapath)

    def load_from_file(self, file_path) -> List[Dict]:
        '''
        Loads examples from CSV file
        :param file_path:
        :return:
        '''
        df = pandas.read_csv(file_path, sep=',', encoding='utf-8-sig')
        return df.to_dict('records')
    def get_examples(self):
        '''
        It needs to only have attributes contained in the specified prompt
        :param type:
        :return:
        '''
        examples = [
            {
                'question': example['question'],
                'answer': example['answer'],
            }
            for example in self.examples
        ]
        return examples

    def get_prompt(self, question: str):
        '''
        Generates a prompt for the message generation task
        :param question:
        :return:
        '''
        prompt = self.select_examples(
            input_variables=["question","answer"],
            selection_strategy="max_marginal",
            prefix=""""
            Verify if given customer question is relevant to the use case mentioned in system message.
            Answer with only 'yes' or 'no'.
            """,
            suffix="""
            Customer question: {question}, Is question relevant: {answer}
            """,
            num_examples=5

        )
        print('---------------------------')
        print(prompt)

        prompt = prompt.format(
            question=question,
            answer=""
        )
        return prompt

    def get_max_marginal_selector(self, examples : List[Dict], k: int):
        '''
        Returns a MaxMarginalRelevanceExampleSelector
        :param examples:
        :param k:
        :return:
        '''
        print(examples)
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            # This is the list of examples available to select from.
            examples,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            OpenAIEmbeddings(),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=k
        )
        return example_selector
    def select_examples(self,selection_strategy:str,
                        input_variables: List[str],
                        prefix: str, suffix: str,
                        num_examples: int = 2) -> FewShotPromptTemplate:
        '''
        Returns a FewShotPromptTemplate
        :param selection_strategy:
        :param input_variables:
        :param prefix:
        :param suffix:
        :param num_examples:
        :return:
        '''

        if os.name == 'nt':
            os.environ['REQUESTS_CA_BUNDLE'] = "C:/python/openai/openai.crt"

        examples = self.get_examples()
        example_prompt = PromptTemplate(
            input_variables=input_variables,
            template=suffix,
        )

        if selection_strategy == "max_marginal":
            example_selector = self.get_max_marginal_selector(examples=examples,
                                                              k=num_examples)
        else:
            raise ValueError(f"{selection_strategy} example selection strategy is not implemented")

        similar_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
        )
        return similar_prompt


