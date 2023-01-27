import pathlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence

import pandas as pd
from tqdm.auto import tqdm

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class NerTags(ABC):
    @property
    @abstractmethod
    def _tags(self) -> list[str]:
        raise NotImplementedError

    @property
    def _delimiter(self) -> str:
        return "|"

    @property
    def instruction(self) -> str:
        return (
            f"Predict the named entity types for each token"
            f" (delimited by '{self._delimiter}')."
            f" The possible types are: {' '.join(self._tags)}."
        )

    def build_answer(self, tokens: Sequence[str], tags: Sequence[str]) -> str:
        return (
            f"Sentence: {self._delimiter.join(tokens)}\n\n"
            f"Named Entity Types: {self._delimiter.join(tags)}\n\n"
        )


class Ell4Tags(NerTags):
    @property
    def _tags(self) -> list[str]:
        tags = ["O"]  # outside
        for position in ["B", "E", "I", "S"]:
            for type_ in ["LOC", "MISC", "ORG", "PERSON"]:
                tags.append(f"{position}-{type_}")  # E.g. B-PERSON
        # Sanity checks
        assert "O" in tags
        assert "B-PERSON" in tags

        return tags


class Ell18Tags(NerTags):
    @property
    def _tags(self) -> list[str]:
        tags = ["O"]  # outside
        for position in ["B", "E", "I", "S"]:
            for type_ in ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE",
                          "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG",
                          "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME",
                          "WORK_OF_ART"]:
                tags.append(f"{position}-{type_}")  # E.g. E-FAC
        # Sanity checks
        assert "O" in tags
        assert "E-FAC" in tags
        return tags


def group_by_sentence(rows: Iterable) -> Iterator[tuple[list[str], list[str]]]:
    # Sentence ID is empty except first word of sentence.
    tokens, tags = [], []
    for _, row in rows:
        if row["Sent_ID"]:
            yield tokens, tags
            tokens, tags = [], []  # Reset.
        tokens.append(row["Word"])
        tags.append(row["Tag"])
    if tokens and tags:  # Don't yield empty final sentence.
        yield tokens, tags


class GreekNER(AbstractDataset):
    """This class should not be instantiated; instead
    instantiate a subclass that defines `self._path`."""

    def __init__(self, name: str, source: str, tags: NerTags) -> None:
        super().__init__(name, source)
        self._tags = tags
        self._path = None

    def get_data(self) -> None:
        df = pd.read_csv(
            self._path, header=0, names=["Sent_ID", "Word", "_", "Tag"])
        task_type = TASK_TYPE.NAMED_ENTITY_RECOGNITION
        jurisdiction = JURISDICTION.GREECE
        prompt_language = "en"
        answer_language = "el"  # TODO: following GermanLER here; it's actually a structured representation though...

        introduction_sentence = "Consider the following sentence in Greek."
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


class Ell18Dataset(GreekNER):
    def __init__(self):
        super().__init__(
            "Ell18 Greek NER",
            "https://github.com/nmpartzio/elNER/blob/master/dataset/elNER18/elNER18_iobes.csv",
            Ell18Tags())
        self._path = pathlib.Path("raw_data/elNER18_iobes.csv")


class Ell4Dataset(GreekNER):
    def __init__(self):
        super().__init__(
            "Ell4 Greek NER",
            "https://github.com/nmpartzio/elNER/blob/master/dataset/elNER4/elNER4_iobes.csv",
            Ell4Tags())
        self._path = pathlib.Path("raw_data/elNER4_iobes.csv")
