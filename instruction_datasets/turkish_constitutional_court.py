from datasets import load_dataset

import instruction_manager
from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import multiple_choice

_BLANK_INSTRUCTION = ''
_MC_OPTIONS = ["Violation", "No violation"]


class TurkishConstitutionalCourt(AbstractDataset):

    def __init__(self):
        super().__init__(
            "TurkishConstitutionalCourt",
            "https://huggingface.co/datasets/KocLab-Bilkent/turkish-constitutional-court")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset('KocLab-Bilkent/turkish-constitutional-court', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.TURKEY
        prompt_language = "tr"
        answer_language = "en"

        for example in df:
            subset = "turkish_constitutional_violation_no_violation"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Case Description: {example['Text']}"
            answer = f"Judgement: {example['Label']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction, subset)

        for example in df:
            task_type = TaskType.MULTIPLE_CHOICE
            subset = "turkish_constitutional_multiple_choice"
            outcomes = multiple_choice.sample_markers_for_options(_MC_OPTIONS)
            assert len(outcomes) == 2
            outcome_mc1 = outcomes[0 if example["Label"] == "No violation" else 1]
            text = example['Text']
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Question: {text} How would the court find?\n" \
                     f"{outcomes[0]} The court should find No violation.\n{outcomes[1]} The court should find Violation."
            answer = f"Answer: {outcome_mc1}."  # e.g. "Answer: (a)."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction, subset)

            outcome_mc1 = outcomes[1 if example["Label"] == "No violation" else 0]
            text = example['Text']
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Question: {text} How would the court find?\n" \
                     f"{outcomes[0]} The court should find Violation.\n{outcomes[1]} The court should find No violation."
            answer = f"Answer: {outcome_mc1}."  # e.g. "Answer: (a)."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction, subset)
