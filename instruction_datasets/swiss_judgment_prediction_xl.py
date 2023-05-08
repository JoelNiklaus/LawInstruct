from datasets import load_dataset

import instruction_manager
from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


class SwissJudgmentPredictionXL(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissJudgmentPredictionXL",
            "https://huggingface.co/datasets/rcds/swiss_judgment_prediction_xl")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset('rcds/swiss_judgment_prediction_xl', 'full', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        answer_language = "en"
        for example in df:
            instructions_group = 'swiss_judgment_dismiss_approve'
            instruction, instruction_language = instructions.sample(instructions_group)
            answer = f"Judgement: {example['label']}"

            if len(example['facts']) > 100:
                prompt = f"Facts: {example['facts']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

            if len(example['considerations']) > 100:
                prompt = f"Considerations: {example['considerations']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)
