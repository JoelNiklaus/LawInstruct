import pandas as pd
from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class LegalQA(AbstractDataset):
    def __init__(self):
        super().__init__("LegalQA", "https://github.com/siatnlp/LegalQA")

    def get_data(self):
        df = pd.read_csv(f"{self.raw_data_dir}/LegalQA-all-train.csv")

        df = df[df['label'] == 1]

        instruction_bank = [
            "Answer the following question according to Chinese law, use plain language as if you are a lawyer answering on an online forum.",
            "This is a question on a Chinese online forum for legal advice. Do not cite case law and use plain language.",
            "Answer the question as a lawyer according to Chinese law, be informal."]
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.CHINA
        prompt_language = "en"

        for q, a in zip(df['question: body'], df['answer']):
            text = f"{self.random.choice(instruction_bank)}\n\nQ:{q}\nA:{a}"
            yield self.build_data_point(prompt_language, "zh", text, task_type, jurisdiction)
