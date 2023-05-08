from datasets import load_dataset

import instruction_manager
from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


class SwissCriticalityPrediction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissCriticalityPrediction",
            "https://huggingface.co/datasets/rcds/swiss_criticality_prediction")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        answer_language = "en"

        df = load_dataset('rcds/swiss_criticality_prediction', 'full', split='train')
        for example in df:
            instruction, instruction_language = instructions.sample("swiss_judgment_criticality")
            answer = f"Criticality: {example['citation_label']}"

            if len(example['facts']) > 100:
                prompt = f"Facts: {example['facts']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

            if len(example['considerations']) > 100:
                prompt = f"Considerations: {example['considerations']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)
