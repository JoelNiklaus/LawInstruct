from datasets import load_dataset

import instruction_manager
from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


class SwissLeadingDecisions(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissLeadingDecisions",
            "https://huggingface.co/datasets/rcds/swiss_leading_decision_summarization")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.SUMMARIZATION
        jurisdiction = Jurisdiction.SWITZERLAND

        df = load_dataset('rcds/swiss_leading_decision_summarization', 'full', split='train')
        for example in df:
            if example['text'] and example['regeste']:
                subset = "swiss_leading_decision_summarization"
                instruction, instruction_language = instructions.sample(subset)
                answer = f"Summary (Regeste): {example['regeste']}"

                prompt = f"Swiss Leading Decision: {example['text']}"
                yield self.build_data_point(instruction_language, example["language"], example["language"],
                                            instruction, prompt, answer, task_type, jurisdiction, subset)

