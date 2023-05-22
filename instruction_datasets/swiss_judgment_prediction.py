from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager
import multiple_choice

_BLANK_INSTRUCTION = ''
# TODO(joel): Why does the language not quite match between these two?
_MC_OPTIONS_1 = ["The court should dismiss the case.", "The court should affirm the case."]
_MC_OPTIONS_2 = ["The court should approve the case.", "The court should dismiss the case."]


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
            markers = multiple_choice.sample_markers_for_options(_MC_OPTIONS_1)
            outcome_mc1 = markers[example["label"]]
            text = example['text']
            subset = "swiss_judgment_multiple_choice"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Question: {text} How would the court find?\n{markers[0]} The court should dismiss the case.\n{markers[1]} The court should affirm the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, example["language"],
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction, subset)

            markers = multiple_choice.sample_markers_for_options(_MC_OPTIONS_2)
            outcome_mc1 = list(reversed(markers))[example["label"]]
            text = example['text']
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Question: {text} How would the court find?\n{markers[0]} The court should approve the case.\n{markers[1]} The court should dismiss the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, example["language"],
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction, subset)
