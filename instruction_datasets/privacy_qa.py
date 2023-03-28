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

        for index, example in df.iterrows():
            text = f"Determine if the term mentioned from the privacy policy is relevant or irrelevant to the given question.\n\n" \
                   f"Q: {example['Query']}\n" \
                   f"Term: {example['Segment']}\n" \
                   f"A: {example['Label']}"
            yield self.build_data_point(prompt_language, "en", text, task_type,
                                        jurisdiction)
