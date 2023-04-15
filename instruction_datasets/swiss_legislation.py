from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
from instruction_datasets.swiss_rulings import get_canton_name

_BLANK_INSTRUCTION = ''


class SwissLegislation(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissLegislation",
            "https://huggingface.co/datasets/rcds/swiss_legislation")

    def get_data(self):
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        instruction_language = 'en'
        answer_language = "en"

        df = load_dataset('rcds/swiss_legislation', 'full', split='train')
        for example in df:
            prompt = f"Law: {example['pdf_content']}"

            if example['canton']:
                instruction = "Where do you think this law was passed?"
                answer = f"Canton: {get_canton_name(example['canton'])}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

            if example['title']:
                instruction = "What do you think is the official long title of this law?"
                answer = f"Title: {example['title']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

            if example['short']:
                instruction = "What do you think is the short title of this law?"
                answer = f"Short Title: {example['short']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

            if example['abbreviation']:
                instruction = "What do you think is the abbvreviation of this law?"
                answer = f"Abbreviation: {example['abbreviation']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)