from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


def get_multiple_choice_instruction_bank():
    return [
        'Please answer these multiple choice questions. Denote the correct answer as "Answer".',
        "Pick the most likely correct answer."
    ]


class TurkishConstitutionalCourt(AbstractDataset):

    def __init__(self):
        super().__init__(
            "TurkishConstitutionalCourt",
            "https://huggingface.co/datasets/KocLab-Bilkent/turkish-constitutional-court")

    def get_data(self):
        df = load_dataset('KocLab-Bilkent/turkish-constitutional-court', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.TURKEY
        instruction_language = 'en'
        prompt_language = "tr"
        answer_language = "en"
        for example in df:
            instruction = f"Determine if you think the Turkish Court of Cassation will label the case description " \
                          f"as Violation or No Violation."
            prompt = f"Case Description: {example['Text']}"
            answer = f"Judgement: {example['Label']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction)

            task_type = TaskType.MULTIPLE_CHOICE
            outcome_mc1 = ["(a)", "(b)"][0 if example["Label"] == "No violation" else 1]
            text = example['Text']
            instruction = self.random.choice(get_multiple_choice_instruction_bank())
            prompt = f"Question: {text} How would the court find?\n" \
                     f"(a) The court should find No violation.\n(b) The court should find Violation."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction)

            outcome_mc1 = ["(b)", "(a)"][0 if example["Label"] == "No violation" else 1]
            text = example['Text']
            instruction = self.random.choice(get_multiple_choice_instruction_bank())
            prompt = f"Question: {text} How would the court find?\n" \
                     f"(a) The court should find Violation.\n(b) The court should find No violation."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction,
                                        prompt, answer, task_type, jurisdiction)
