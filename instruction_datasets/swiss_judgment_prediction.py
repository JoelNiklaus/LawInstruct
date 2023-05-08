from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

_BLANK_INSTRUCTION = ''


class SwissJudgmentPrediction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissJudgmentPrediction",
            "https://huggingface.co/datasets/rcds/swiss_judgment_prediction")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset('rcds/swiss_judgment_prediction', 'all+mt', split='train')
        task_type = TaskType.MULTIPLE_CHOICE
        jurisdiction = Jurisdiction.SWITZERLAND
        answer_language = "en"

        for example in df:
            outcome_mc1 = ["(a)", "(b)"][example["label"]]
            text = example['text']
            instruction, instruction_language = instructions.sample("swiss_judgment_multiple_choice")
            prompt = f"Question: {text} How would the court find?\n(a) The court should dismiss the case.\n(b) The court should affirm the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, example["language"],
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction)

            outcome_mc1 = ["(b)", "(a)"][example["label"]]
            text = example['text']
            instruction, instruction_language = instructions.sample("swiss_judgment_multiple_choice")
            prompt = f"Question: {text} How would the court find?\n(a) The court should approve the case.\n(b) The court should dismiss the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, example["language"],
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction)
