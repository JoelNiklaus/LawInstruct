from collections.abc import Iterator
import json
import pathlib

from tqdm.auto import tqdm

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

from .greek_ner import NerTags


class IndianNerTags(NerTags):

    @property
    def _tags(self) -> list[str]:
        # O wasn't in the original dataset, but we need for our representation.
        return [
            'O', 'COURT', 'PETITIONER', 'RESPONDENT', 'JUDGE', 'LAWYER', 'DATE',
            'ORG', 'GPE', 'STATUTE', 'PROVISION', 'PRECEDENT', 'CASE_NUMBER',
            'WITNESS', 'OTHER_PERSON'
        ]


def find_sub_list(needle: list, haystack: list) -> tuple[int, int]:
    needle_length = len(needle)
    for ind in (i for i, e in enumerate(haystack) if e == needle[0]):
        if haystack[ind:ind + needle_length] == needle:
            return ind, ind + needle_length
    return 0, 0  # Empty slice.


class IndianNER(AbstractDataset):

    def __init__(self):
        super().__init__("IndianNER",
                         "https://github.com/Legal-NLP-EkStep/legal_NER")
        self._tags = IndianNerTags()
        self._path = pathlib.Path(
            f"{self.raw_data_dir}/NER_TRAIN_JUDGEMENT.json")

    def get_data(self, instructions: instruction_manager.InstructionManager) -> Iterator[dict]:
        task_type = TaskType.NAMED_ENTITY_RECOGNITION
        jurisdiction = Jurisdiction.INDIA
        instruction_language = "en"
        prompt_language = "en"
        answer_language = "ner"

        introduction_sentence = "Consider the following sentence in English."
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

                start, end = find_sub_list(name, tokens)
                for i in range(start, end):
                    tags[i] = label

            instruction = self.random.choice(instruction_bank)
            prompt, answer = self._tags.build_answer(tokens, tags)
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)
