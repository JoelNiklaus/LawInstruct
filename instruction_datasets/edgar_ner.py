import pathlib
from collections.abc import Iterable, Iterator

import pandas as pd
from tqdm.auto import tqdm

from abstract_dataset import AbstractDataset, TASK_TYPE, JURISDICTION
from .greek_ner import NerTags


class EdgarTags(NerTags):
    @property
    def _tags(self) -> list[str]:
        tags = ["O"]  # outside
        for position in ["B", "I"]:
            for type_ in ["BUSINESS",
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
        super().__init__("EDGAR", "https://github.com/terenceau2/E-NER-Dataset/blob/main/all.csv")
        self._tags = EdgarTags()
        self._path = pathlib.Path("raw_data/all.csv")

    def get_data(self) -> Iterator[dict]:
        df = pd.read_csv(self._path, header=None, names=["Word", "Tag"], na_filter=False)
        task_type = TASK_TYPE.NAMED_ENTITY_RECOGNITION
        jurisdiction = JURISDICTION.GREECE
        prompt_language = "en"
        answer_language = "en"  # TODO: following GermanLER here; it's actually a structured representation though...

        introduction_sentence = "Consider the following English sentence from the United States SEC."
        instruction_bank = [
            introduction_sentence + " " + self._tags.instruction
        ]

        for tokens, tags in group_by_sentence(tqdm(df.iterrows(), total=len(df))):
            text = (
                f"{self.random.choice(instruction_bank)}\n\n"
                f"{self._tags.build_answer(tokens, tags)}"
            )
            yield self.build_data_point(
                prompt_language, answer_language, text, task_type, jurisdiction)

