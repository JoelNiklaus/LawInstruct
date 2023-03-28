import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class MCExamsLaw(AbstractDataset):

    def __init__(self):
        # TODO do we have an url here: hand built by peter
        super().__init__("MCExamsLaw", "mc_exams_law")

    def get_data(self):
        df = pd.read_csv(
            f"{self.raw_data_dir}/raw_legal_mc_with_explanations.csv")
        task_type = TaskType.MULTIPLE_CHOICE
        jurisdiction = Jurisdiction.US
        prompt_language = "en"

        instruction_bank = [
            "Answer these questions according to the laws of the United States.",
            "Pick the best answer according to U.S. law.",
            "Pick the correct multiple choice answer according to American law."
        ]
        instruction_bank_expl = [
            "Answer these questions according to the laws of the United States. First explain your answer.",
            "Pick the best answer according to U.S. law. First explain your answer.",
            "Pick the correct multiple choice answer according to American law. Explain your answer then give the correct choice."
        ]
        for idx, row in df.iterrows():
            q, a, explanation, source = row["Question"], row["Answer"], row[
                "Explanation"], row["Source"]

            # No chain of thought
            text = f"{self.random.choice(instruction_bank)}\n\nQ:{q}\nA:{a}"
            yield self.build_data_point(prompt_language, "en", text, task_type,
                                        jurisdiction)

            # Chain of thought
            text = f"{self.random.choice(instruction_bank_expl)}\n\nQ:{q}\nExplanation: {explanation}\nA:{a}"
            yield self.build_data_point(prompt_language, "en", text, task_type,
                                        jurisdiction)
