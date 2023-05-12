from datasets import load_dataset

import instruction_manager
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

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        answer_language = "en"

        df = load_dataset('rcds/swiss_legislation', 'full', split='train')
        for example in df:
            prompt = f"Law: {example['pdf_content']}"

            if example['canton']:
                subset = "swiss_legislation_canton"
                instruction, instruction_language = instructions.sample(subset)
                answer = f"Canton: {get_canton_name(example['canton'])}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction, subset)

            if example['title']:
                subset = "swiss_legislation_title"
                instruction, instruction_language = instructions.sample(subset)
                answer = f"Title: {example['title']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction, subset)

            if example['short']:
                subset = "swiss_legislation_short"
                instruction, instruction_language = instructions.sample(subset)
                answer = f"Short Title: {example['short']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction, subset)

            if example['abbreviation']:
                subset = "swiss_legislation_abbreviation"
                instruction, instruction_language = instructions.sample(subset)
                answer = f"Abbreviation: {example['abbreviation']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction, subset)
