from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
from instruction_datasets.swiss_rulings import get_canton_name

_BLANK_INSTRUCTION = ''


class SwissJudgmentPredictionXL(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissJudgmentPredictionXL",
            "https://huggingface.co/datasets/rcds/swiss_judgment_prediction_xl")

    def get_data(self):
        df = load_dataset('rcds/swiss_judgment_prediction', 'full', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        instruction_language = 'en'
        answer_language = "en"
        for example in df:
            judgment = ["dismiss", "approve"][example['label']]
            instruction = f"Determine if you think the Swiss court will dismiss or approve the case."
            answer = f"Judgement: {judgment}"

            prompt = f"Facts: {example['facts']}"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction, prompt, answer, task_type, jurisdiction)

            prompt = f"Considerations: {example['considerations']}"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction, prompt, answer, task_type, jurisdiction)
