import json
import pathlib
import re
from collections.abc import Iterator

from abstract_dataset import AbstractDataset, TASK_TYPE, JURISDICTION


_CATEGORIES: dict[str, str] = {
    "PREAMBLE": "Preamble",
    "FAC": "Facts",
    "RLC": "Ruling by lower court",
    "ISSUE": "Issues",
    "ARG_PETITIONER": "Argument by Petitioner",
    "ARG_RESPONDENT": "Argument by Respondent",
    "ANALYSIS": "Analysis",
    "STA": "Statute",
    "PRE_RELIED": "Precedent Relied",
    "PRE_NOT_RELIED": "Precedent Not Relied",
    "RATIO": "Ratio of the decision",
    "RPC": "Ruling by Present Court",
    "NONE": "Nothing meaningful",
}


class IndianTextSegmentation(AbstractDataset):
    def __init__(self):
        super().__init__(
            "BUILD",
            "https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline")
        self._path = pathlib.Path("raw_data/indian_text_segmentation.json")

    def get_data(self) -> Iterator[dict]:
        task_type = TASK_TYPE.TEXT_CLASSIFICATION
        jurisdiction = JURISDICTION.INDIA
        prompt_language = "en"
        answer_language = "hi"

        with open(self._path, "r") as f:
            data = json.load(f)
        for passage in data:
            spans = passage["annotations"][0]["result"]
            for span in spans:
                raw_passage = span["value"]["text"]
                passage = re.sub(
                    r'\W+', ' ', raw_passage)  # Collapse whitespace.
                label = span["value"]["labels"][0]

                text = (
                    "In Indian case law, what is the rhetorical role of this part of a court judgment?"
                    f" The options are {', '.join(list(_CATEGORIES.values()))}."
                    f"\n\nPassage: {passage}"
                    f"\n\nRole: {_CATEGORIES[label]}"
                )

                yield self.build_data_point(
                    prompt_language, answer_language, text, task_type, jurisdiction
                )