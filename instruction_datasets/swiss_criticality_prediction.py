from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


class SwissCriticalityPrediction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissCriticalityPrediction",
            "https://huggingface.co/datasets/rcds/swiss_criticality_prediction")

    def get_data(self):
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        instruction_language = 'en'
        answer_language = "en"

        df = load_dataset('rcds/swiss_criticality_prediction', 'full', split='train')
        for example in df:
            instruction = "How important or critical is this case? " \
                          "The case can be non-critical " \
                          "or range from critical-1 (least critical) to critical-4 (most critical)."
            answer = f"Criticality: {example['citation_label']}"

            prompt = f"Facts: {example['facts']}"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction, prompt, answer, task_type, jurisdiction)

            prompt = f"Considerations: {example['considerations']}"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction, prompt, answer, task_type, jurisdiction)
