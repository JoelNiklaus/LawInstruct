from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
from instruction_datasets.swiss_rulings import get_canton_name

_BLANK_INSTRUCTION = ''


class SwissJudgmentPrediction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissJudgmentPrediction",
            "https://huggingface.co/datasets/rcds/swiss_judgment_prediction")

    def get_data(self):
        df = load_dataset('rcds/swiss_judgment_prediction', 'all+mt', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        instruction_language = 'en'
        answer_language = "en"
        for example in df:
            court_location = "" if example['canton'] == "n/a" \
                else f"The lower court is located in {get_canton_name(example['canton'])}."

            multiple_choice_instruction_bank = [
                'Please answer these multiple choice questions. Denote the correct answer as "Answer".',
                "Pick the most likely correct answer."
            ]
            instruction = self.random.choice(multiple_choice_instruction_bank)

            task_type = TaskType.MULTIPLE_CHOICE
            outcome_mc = ["(a)", "(b)"][example["label"]]
            prompt = f"Question: {example['text']} How would the Federal Swiss Supreme court find? {court_location}\n" \
                     f"(a) The court should dismiss the case.\n(b) The court should affirm the case."
            answer = f"Answer: {outcome_mc}."
            yield self.build_data_point(instruction_language, answer_language, example["language"],
                                        instruction, prompt, answer, task_type, jurisdiction)

            outcome_mc = ["(b)", "(a)"][example["label"]]
            prompt = f"Question: {example['text']} How would the Federal Swiss Supreme court find? {court_location}\n" \
                     f"(a) The court should approve the case.\n(b) The court should dismiss the case."
            answer = f"Answer: {outcome_mc}."
            yield self.build_data_point(instruction_language, answer_language, example["language"],
                                        instruction, prompt, answer, task_type, jurisdiction)
