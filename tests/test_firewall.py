from trinity.prompt_firewall import PromptFirewall
import pandas as pd
#import confusion_matrix
from sklearn.metrics import confusion_matrix
firewall = PromptFirewall()
def calculate_confusion_matrix():
    # Load data
    df = pd.read_csv("data/firewall.csv")
    #randomly shuffle data
    df = df.sample(frac=1)
    #split data into train and test
    train = df.iloc[:int(0.7 * len(df))]
    test = df.iloc[int(0.7 * len(df)):]
    #rows to dict
    train = train.to_dict('records')
    test = test.to_dict('records')
    y_true = [example['answer'] for example in test]
    y_pred = []
    for example in test:
        result = firewall.verify_question(example['question'])
        y_pred.append(result)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=["yes", "no"]
    )
    print(matrix)

if __name__ == '__main__':
    calculate_confusion_matrix()