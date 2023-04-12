from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


class SwissLawAreaPrediction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissLawAreaPrediction",
            "https://huggingface.co/datasets/rcds/swiss_law_area_prediction")

    def get_data(self):
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        instruction_language = 'en'
        answer_language = "en"

        df = load_dataset('rcds/swiss_law_area_prediction', 'main', split='train')
        for example in df:
            instruction = "What main area of law is this case related to?"
            prompt = f"Facts: {example['facts']}"
            answer = f"Area of Law: {example['label']} Law"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction, prompt, answer, task_type, jurisdiction)

        for law_area in ['civil', 'criminal', 'public']:
            df = load_dataset('rcds/swiss_law_area_prediction', law_area, split='train')
            for example in df:
                instruction = f"What sub-area of {law_area} law is this case related to?"
                prompt = f"Facts: {example['facts']}"
                answer = f"Sub-Area of Law: {example['label']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)