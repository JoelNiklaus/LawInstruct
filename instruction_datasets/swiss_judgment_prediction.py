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


class SwissJudgmentPrediction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissJudgmentPrediction",
            "https://huggingface.co/datasets/swiss_judgment_prediction")

    def get_data(self):
        df = load_dataset('swiss_judgment_prediction', 'all+mt', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        prompt_language = "en"
        for example in df:
            court_location = "" if example[
                'region'] == "n/a" else f"The court is located in {example['region']}."
            judgement = ["dismiss", "approve"][example['label']]
            instruction = f"Determine if you think the Swiss court will dismiss or approve the case. {court_location}"
            text = f"Facts:{example['text']}\nJudgement: {judgement}"
            yield self.build_data_point(prompt_language, example["language"],
                                        instruction, text, task_type,
                                        jurisdiction)

            instruction = "What area of law is this case related to?"
            text = f"Case:{example['text']}\nArea of Law: {example['legal area']}"
            yield self.build_data_point(prompt_language, example["language"],
                                        instruction, text, task_type,
                                        jurisdiction)

            if court_location != "":
                instruction = "Where do you think this case was adjudicated?"
                text = f"Case:{example['text']}\nRegion: {example['region']}"
                yield self.build_data_point(prompt_language,
                                            example["language"], instruction,
                                            text, task_type, jurisdiction)

            task_type = TaskType.MULTIPLE_CHOICE
            outcome_mc1 = ["(a)", "(b)"][example["label"]]
            text = example['text']
            instruction = self.random.choice(
                get_multiple_choice_instruction_bank())
            text = f"Question: {text} How would the court find?\n(a) The court should dismiss the case.\n(b) The court should affirm the case.\n" \
                   f"Answer: {outcome_mc1}."
            yield self.build_data_point(prompt_language, example["language"],
                                        instruction, text, task_type,
                                        jurisdiction)

            outcome_mc1 = ["(b)", "(a)"][example["label"]]
            text = example['text']
            instruction = self.random.choice(
                get_multiple_choice_instruction_bank())
            text = f"Question: {text} How would the court find?\n(a) The court should approve the case.\n(b) The court should dismiss the case.\n" \
                   f"Answer: {outcome_mc1}."
            yield self.build_data_point(prompt_language, example["language"],
                                        instruction, text, task_type,
                                        jurisdiction)
