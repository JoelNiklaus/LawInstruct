from bs4 import BeautifulSoup
import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class StackExchangeQuestionsLegal(AbstractDataset):

    def __init__(self):
        super().__init__("StackExchangeQuestionsLegal",
                         "https://law.stackexchange.com/")

    def get_data(self):
        # Legal Stack Exchange questions are usually high quality

        df = pd.read_csv(f"{self.raw_data_dir}/stack-exchange.csv")
        instruction_bank = [
            "Answer the following legal question. Cite relevant evidence when possible.",
            "Answer this online forum question about the law.",
            "Provide an explanation for this short form legal question."
        ]
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.UNKNOWN
        prompt_language = "en"

        for idx, example in df.iterrows():
            soup = BeautifulSoup(example["body"])
            text = soup.get_text()
            question = text
            soup = BeautifulSoup(example["body.1"])
            text = soup.get_text()
            answer = text
            instruction = f"{self.random.choice(instruction_bank)}"
            if self.random.random() > .7:
                instruction += " " + f"This question is about: {','.join([x.replace('>', '').replace('<', '').replace('-', ' ').strip() for x in example['tags'].split('>') if x.replace('>', '').replace('<', '').strip() != ''])}."

            text = f"Question: {question}\nAnswer: {answer}"
            yield self.build_data_point(prompt_language, "en", instruction,
                                        text, task_type, jurisdiction)
