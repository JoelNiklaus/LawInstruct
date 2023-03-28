import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class LegalQA(AbstractDataset):

    def __init__(self):
        super().__init__("LegalQA", "https://github.com/siatnlp/LegalQA")

    def get_data(self):
        df = pd.read_csv(f"{self.raw_data_dir}/LegalQA-all-train.csv")

        df = df[df['label'] == 1]

        instruction_bank = [
            "Answer the following question according to Chinese law, use plain language as if you are a lawyer answering on an online forum.",
            "This is a question on a Chinese online forum for legal advice. Do not cite case law and use plain language.",
            "Answer the question as a lawyer according to Chinese law, be informal."
        ]
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.CHINA
        prompt_language = "en"

        for q, a in zip(df['question: body'], df['answer']):
            instruction = self.random.choice(instruction_bank)
            text = f"Q:{q}\nA:{a}"
            yield self.build_data_point(prompt_language, "zh", instruction,
                                        text, task_type, jurisdiction)
