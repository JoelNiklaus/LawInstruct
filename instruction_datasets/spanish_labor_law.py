import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class SpanishLaborLaw(AbstractDataset):

    def __init__(self):
        super().__init__("SpanishLaborLaw",
                         "https://zenodo.org/record/4256718#.Y5PoC7LMIlg")

    def get_data(self):
        df = pd.read_csv(f"{self.raw_data_dir}/spanish_legal_qa.csv")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.SPAIN
        prompt_language = "en"

        instruction_bank = [
            "Consider this Spanish Labor Law translated passage. Answer the question using an extractive snippet of text.",
            "Consider this Spanish Labor Law translated passage. Answer the question from the context.",
            "Answer the following Spanish labor law question given the legal provision."
        ]
        for idx, row in df.iterrows():
            question, context, answer = row["Question"], row["context"], row[
                "Answer text"]
            text = f"{self.random.choice(instruction_bank)}\n\nContext: {context}\nQ: {question}\nA: {answer}"
            yield self.build_data_point(prompt_language, "es", text, task_type,
                                        jurisdiction)
