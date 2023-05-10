from collections.abc import Iterator
import pathlib

import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

_TEXT4LABEL = {
    "bad": "risky",
    "neutral": "non-risky",
}


class PrivacySummarization(AbstractDataset):

    def __init__(self):
        super().__init__(
            "PrivacySummarization",
            "https://github.com/senjed/Summarization-of-Privacy-Policies/blob/master/TOSDR_full_content_au_labeled_v2.csv"
        )
        self._path = pathlib.Path(
            f"{self.raw_data_dir}/TOSDR_full_content_au_labeled_v2.csv")

    def get_data(self, instructions: instruction_manager.InstructionManager) -> Iterator[dict]:
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.UNKNOWN
        instruction_language: str
        prompt_language = "en"
        answer_language = "en"

        df = pd.read_csv(self._path, header=0)
        for _, record in df.iterrows():
            # `QouteText` is a typo in the original dataset.
            passage, label = record["QouteText"], record["Point"]
            instruction, instruction_language = instructions.sample("privacy_summarization")
            prompt = passage
            answer = _TEXT4LABEL[label]
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)
