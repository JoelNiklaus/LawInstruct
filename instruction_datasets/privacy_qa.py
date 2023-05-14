import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class PrivacyQA(AbstractDataset):

    def __init__(self):
        super().__init__(
            "PrivacyQA",
            "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = pd.read_csv(f"{self.raw_data_dir}/policy_train_data.csv", sep="\t")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.UNKNOWN
        instruction_language: str
        prompt_language = "en"
        subset = "privacy_qa"
        instruction, instruction_language = instructions.sample(subset)

        for index, example in df.iterrows():
            prompt = f"Q: {example['Query']}\nTerm: {example['Segment']}"
            answer = f"A: {example['Label']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)
