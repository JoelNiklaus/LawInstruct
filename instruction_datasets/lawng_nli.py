from collections.abc import Iterable

import pandas as pd

from abstract_dataset import AbstractDataset
from abstract_dataset import DataPoint
from abstract_dataset import JURISDICTION
from abstract_dataset import TASK_TYPE

_PREMISE_COLS = (
    "long_premise",
    "short_premise",
    "long_premise_with_citations",
    "short_premise_with_citations",
)


class LawngNli(AbstractDataset):
    """The LawngNLI dataset.

  Reads the LawngNLI dataset locally.
  """

    def __init__(self):
        super().__init__("LawngNli", "https://github.com/wbrun0/LawngNLI")

    def get_data(self) -> Iterable[DataPoint]:
        """Returns the data points in the dataset.

    Returns:
      An iterable of data points.
    """
        file = f"{self.raw_data_dir}/LawngNLI.xz"
        # They decided to use a pickle file instead of a csv file.
        # Now we're forced to read a pickled pandas dataframe.
        df = pd.read_pickle(file)
        jurisdiction = JURISDICTION.US
        prompt_language = "en"
        task_type = TASK_TYPE.NATURAL_LANGUAGE_INFERENCE
        instruction_bank = [
            "Consider the following matter from a US legal opinion. Does the first passage entail the second fact?",
            "Are these two passages entailed, contradicting, or neutral?",
            "Respond entailment, contradiction, or neutral to these two passages.",
        ]
        word4label = {
            0.0: "entailment",
            2.0: "contradiction",
            1.0: "neutral",
        }
        label_reversal = {
            "entailment": "contradiction",
            "contradiction": "entailment",
            "neutral": "neutral",
        }

        for i, row in df.iterrows():
            for premise_col in _PREMISE_COLS:
                # Add the datapoint.
                datapoint = f"{self.random.choice(instruction_bank)}\n\n" \
                            f"Passage 1: {row[premise_col]}\n" \
                            f"Sentence 2: {row['hypothesis']}\n" \
                            f"Answer: {word4label[row['label']]}"
                yield self.build_data_point(
                    prompt_language, "en", datapoint, task_type, jurisdiction)
                # Add the contradicting datapoint.
                datapoint = f"{self.random.choice(instruction_bank)}\n\n" \
                            f"Passage 1: {row[premise_col]}\n" \
                            f"Sentence 2: {row['contradicted_parenthetical']}\n" \
                            f"Answer: {label_reversal[word4label[row['label']]]}"
                yield self.build_data_point(
                    prompt_language, "en", datapoint, task_type, jurisdiction)