from collections.abc import Iterable

import pandas as pd

from abstract_dataset import AbstractDataset
from abstract_dataset import DataPoint
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

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

    def get_data(self, instructions: instruction_manager.InstructionManager) -> Iterable[DataPoint]:
        """Returns the data points in the dataset.

    Returns:
      An iterable of data points.
    """
        file = f"{self.raw_data_dir}/LawngNLI.xz"
        # They decided to use a pickle file instead of a csv file.
        # Now we're forced to read a pickled pandas dataframe.
        df = pd.read_pickle(file)
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"
        task_type = TaskType.NATURAL_LANGUAGE_INFERENCE
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
                subset = "lawng_nli_entailment"
                instruction, instruction_language = instructions.sample(subset)
                prompt = f"Passage 1: {row[premise_col]}\n" \
                            f"Sentence 2: {row['hypothesis']}\n"
                answer = f"Answer: {word4label[row['label']]}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, subset)
                # Add the contradicting datapoint.
                instruction, instruction_language = instructions.sample(subset)
                prompt = f"Passage 1: {row[premise_col]}\n" \
                            f"Sentence 2: {row['contradicted_parenthetical']}\n"
                answer = f"Answer: {label_reversal[word4label[row['label']]]}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, subset)
