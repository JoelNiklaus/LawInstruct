from datasets import load_dataset

import instruction_manager
from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


class SwissLawAreaPrediction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissLawAreaPrediction",
            "https://huggingface.co/datasets/rcds/swiss_law_area_prediction")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        answer_language = "en"

        df = load_dataset('rcds/swiss_law_area_prediction', 'main', split='train')
        for example in df:
            instruction, instruction_language = instructions.sample("swiss_judgment_area_of_law_main_area")
            answer = f"Area of Law: {example['law_area']} Law"

            if len(example['facts']) > 100:
                prompt = f"Facts: {example['facts']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

            if len(example['considerations']) > 100:
                prompt = f"Considerations: {example['considerations']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)


            instruction, instruction_language = instructions.sample("swiss_judgment_area_of_law_sub_area")
            if example['law_sub_area']:
                answer = f"Sub-Area of Law: {example['law_sub_area']}"

                if len(example['facts']) > 100:
                    prompt = f"Facts: {example['facts']}"
                    yield self.build_data_point(instruction_language, example["language"], answer_language,
                                                instruction, prompt, answer, task_type, jurisdiction)

                if len(example['considerations']) > 100:
                    prompt = f"Considerations: {example['considerations']}"
                    yield self.build_data_point(instruction_language, example["language"], answer_language,
                                                instruction, prompt, answer, task_type, jurisdiction)
