import os

import pandas

from typing import Dict, List
class ExampleDataLoader:
    def __init__(self):
        pass



    def get_folders(self, base_path):
        '''
        Will return list of folders in the base_path
        :param base_path:
        :return:
        '''
        return [f.path for f in os.scandir(base_path) if f.is_dir()]


    def load_example(self, base_folder) -> Dict:
        '''
        Loads example from the folder
        :param base_folder:
        :return:
        '''
        message_f = open(f"{base_folder}/message.txt", 'r', encoding='utf-8')
        message = message_f.read()
        cv_f = open(f"{base_folder}/cv_summary.txt", 'r', encoding='utf-8')
        cv = cv_f.read()
        cv_text_f = open(f"{base_folder}/cv_text.txt", 'r', encoding='utf-8')
        cv_text = cv_text_f.read()
        role_text_f = open(f"{base_folder}/role_text.txt", 'r', encoding='utf-8')
        role_text = role_text_f.read()
        role_f = open(f"{base_folder}/role_summary.txt", 'r', encoding='utf-8')
        role = role_f.read()
        rec_f = open(f"{base_folder}/cv_recommendation.txt", 'r', encoding='utf-8')
        rec = rec_f.read()
        fit_score_f = open(f"{base_folder}/fit_score.txt", 'r', encoding='utf-8')
        fit_score = fit_score_f.read()
        if os.name == 'nt':
            name_surname = base_folder.split('\\')[-1]
        else:
            name_surname = base_folder.split('/')[-1]
        name = name_surname.split('_')[0]
        surname = name_surname.split('_')[1]

        role = Role(description=role_text, summary=role)
        cv = CV(experience=cv_text, summary=cv, name=name, surname=surname)

        ex = {
            'role': role,
            'cv': cv,
            'message': message,
            'rec': rec
        }

        return ex

    def prepare(self, input_folder: str, output_folder: str):
        folders = self.get_folders(input_folder)
        examples = []
        for folder in folders:
            examples.append(self.load_example(folder))

        message_prompts = []
        rec_prompts = []
        target_messages = []
        target_recs = []
        for example in examples:
            target_messages.append(example['message'])
            target_recs.append(example['rec'])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return examples