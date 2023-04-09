from collections.abc import Iterable
from collections.abc import Iterator
import pathlib

import pandas as pd
from tqdm.auto import tqdm

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

from .greek_ner import NerTags


class EdgarTags(NerTags):

    @property
    def _tags(self) -> list[str]:
        tags = ["O"]  # outside
        for position in ["B", "I"]:
            for type_ in [
                "BUSINESS",
                "GOVERNMENT",
                "LEGISLATION/ACT",
                "LOCATION",
                "MISCELLANEOUS",
                "PERSON",
            ]:
                tags.append(f"{position}-{type_}")
        # Sanity checks
        assert "O" in tags
        assert "I-LEGISLATION/ACT" in tags

        return tags


def group_by_sentence(rows: Iterable) -> Iterator[tuple[list[str], list[str]]]:
    # -DOCSTART- as the word separates documents.
    # Blank words separate sentences.
    tokens, tags = [], []
    for _, row in rows:
        if row["Word"] == "-DOCSTART-":
            # Ignore document breaks. We just split on sentences.
            continue
        elif not row["Word"]:
            if tokens and tags:
                yield tokens, tags
            tokens, tags = [], []  # Reset.
        else:
            tokens.append(row["Word"])
            tags.append(row["Tag"])
    if tokens and tags:  # Don't yield empty final sentence.
        yield tokens, tags


class EdgarNER(AbstractDataset):

    def __init__(self):
        super().__init__(
            "EdgarNER",
            "https://github.com/terenceau2/E-NER-Dataset/blob/main/all.csv")
        self._tags = EdgarTags()
        self._path = pathlib.Path(f"{self.raw_data_dir}/edgar_ner.csv")

    def get_data(self) -> Iterator[dict]:
        df = pd.read_csv(self._path,
                         header=None,
                         names=["Word", "Tag"],
                         na_filter=False)
        task_type = TaskType.NAMED_ENTITY_RECOGNITION
        jurisdiction = Jurisdiction.US
        instruction_language = "en"
        prompt_language = "en"
        answer_language = "en"  # TODO: following GermanLER here; it's actually a structured representation though...

        introduction_sentence = "Consider the following English sentence from the United States SEC."
        instruction_bank = [
            introduction_sentence + " " + self._tags.instruction
        ]

        for tokens, tags in group_by_sentence(tqdm(df.iterrows(),
                                                   total=len(df))):
            instruction = self.random.choice(instruction_bank)
            prompt, answer = self._tags.build_answer(tokens, tags)
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)
