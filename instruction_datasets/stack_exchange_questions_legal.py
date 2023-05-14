from bs4 import BeautifulSoup
import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class StackExchangeQuestionsLegal(AbstractDataset):

    def __init__(self):
        super().__init__("StackExchangeQuestionsLegal",
                         "https://law.stackexchange.com/")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        # Legal Stack Exchange questions are usually high quality

        df = pd.read_csv(f"{self.raw_data_dir}/stack-exchange.csv")
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
            instruction: str
            instruction_language: str
            subset = "stack_exchange_questions_legal"
            instruction, instruction_language = instructions.sample(subset)
            if self.random.random() > .7:
                instruction += " " + f"This question is about: {','.join([x.replace('>', '').replace('<', '').replace('-', ' ').strip() for x in example['tags'].split('>') if x.replace('>', '').replace('<', '').strip() != ''])}."

            prompt = f"Question: {question}"
            answer = f"Answer: {answer}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)
