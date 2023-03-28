import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class PrivacyQA(AbstractDataset):

    def __init__(self):
        super().__init__(
            "PrivacyQA",
            "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP")

    def get_data(self):
        df = pd.read_csv(f"{self.raw_data_dir}/policy_train_data.csv", sep="\t")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.UNKNOWN
        prompt_language = "en"
        instruction = "Determining if a term mentioned in a privacy policy is relevant or irrelevant to a given question."

        for index, example in df.iterrows():
            text = f"Q: {example['Query']}\n" \
                   f"Term: {example['Segment']}\n" \
                   f"A: {example['Label']}"
            yield self.build_data_point(prompt_language, "en", instruction,
                                        text, task_type, jurisdiction)
