import pathlib
from collections.abc import Iterable, Iterator
import json

import pandas as pd
from tqdm.auto import tqdm

from abstract_dataset import AbstractDataset, TASK_TYPE, JURISDICTION
from .greek_ner import NerTags


class IndianNerTags(NerTags):
    @property
    def _tags(self) -> list[str]:
        # O wasn't in the original dataset, but we need for our representation.
        return ['O', 'COURT', 'PETITIONER', 'RESPONDENT', 'JUDGE', 'LAWYER', 'DATE',
                'ORG', 'GPE', 'STATUTE', 'PROVISION', 'PRECEDENT',
                'CASE_NUMBER', 'WITNESS', 'OTHER_PERSON']


def find_sub_list(needle: list, haystack: list) -> tuple[int, int]:
    needle_length = len(needle)
    for ind in (i for i, e in enumerate(haystack) if e == needle[0]):
        if haystack[ind:ind+needle_length] == needle:
            return ind, ind+needle_length
    return 0, 0  # Empty slice.


class IndianNER(AbstractDataset):
    def __init__(self):
        super().__init__("IndianNER", "https://github.com/Legal-NLP-EkStep/legal_NER")
        self._tags = IndianNerTags()
        self._path = pathlib.Path("raw_data/NER_TRAIN_JUDGEMENT.json")

    def get_data(self) -> Iterator[dict]:
        task_type = TASK_TYPE.NAMED_ENTITY_RECOGNITION
        jurisdiction = JURISDICTION.INDIA
        prompt_language = "en"
        answer_language = "hi"  # TODO: following GermanLER here; it's actually a structured representation though...

        introduction_sentence = "Consider the following sentence in Greek."
        instruction_bank = [
            introduction_sentence + " " + self._tags.instruction
        ]

        with open(self._path, "r") as f:
            data = json.load(f)
        for sentence in tqdm(data):
            tokens = sentence["data"]["text"].strip().split()
            tags = ["O" for _ in tokens]
            for named_entity in sentence["annotations"][0]["result"]:
                name = named_entity["value"]["text"].strip().split()
                label = named_entity["value"]["labels"][0]
                print(name, label)
                start, end = find_sub_list(name, tokens)
                tags[start:end] = label

            text = (
                f"{self.random.choice(instruction_bank)}\n\n"
                f"{self._tags.build_answer(tokens, tags)}"
            )
            yield self.build_data_point(
                prompt_language, answer_language, text, task_type, jurisdiction
            )