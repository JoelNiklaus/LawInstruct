import pathlib
from collections.abc import Iterator

import pandas as pd

from abstract_dataset import AbstractDataset, TASK_TYPE, JURISDICTION

_TEXT4LABEL = {
    "bad": "risky",
    "neutral": "non-risky",
}


class PrivacySummarization(AbstractDataset):
    def __init__(self):
        super().__init__("PrivacySummarization", "https://github.com/senjed/Summarization-of-Privacy-Policies/blob/master/TOSDR_full_content_au_labeled_v2.csv")
        self._path = pathlib.Path(f"{self.raw_data_dir}/TOSDR_full_content_au_labeled_v2.csv")

    def get_data(self) -> Iterator[dict]:
        task_type = TASK_TYPE.TEXT_CLASSIFICATION
        jurisdiction = JURISDICTION.UNKNOWN
        prompt_language = "en"
        answer_language = "en"

        introduction_sentence = (
            "Is the following statement from a"
            " privacy policy risky or non-risky?"
        )

        df = pd.read_csv(self._path, header=0)
        for _, record in df.iterrows():
            # `QuoteText` is a typo in the original dataset.
            passage, label = record["QouteText"], record["Point"]
            text = (
                f"{introduction_sentence}\n\n"
                f"{passage}\n\n"
                f"{_TEXT4LABEL[label]}"
            )
            yield self.build_data_point(
                prompt_language, answer_language, text, task_type, jurisdiction
            )