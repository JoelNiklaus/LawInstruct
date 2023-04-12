from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


class SwissCourtViewGeneration(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissCourtViewGeneration",
            "https://huggingface.co/datasets/rcds/swiss_court_view_generation")

    def get_data(self):
        task_type = TaskType.TEXT_GENERATION
        jurisdiction = Jurisdiction.SWITZERLAND
        instruction_language = 'en'
        answer_language = "en"

        df = load_dataset('rcds/swiss_court_view_generation', 'full', split='train')
        for example in df:
            instruction = "Write the court's view (considerations) " \
                          "for the given facts description."
            prompt = f"Facts: {example['facts']}"
            answer = f"Court View: {example['considerations']}"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction, prompt, answer, task_type, jurisdiction)

        df = load_dataset('rcds/swiss_court_view_generation', 'origin', split='train')
        for example in df:
            instruction = "Write the court's view (considerations) " \
                          "for the given facts and considerations from the lower court."
            prompt = f"Facts: {example['origin_facts']}\nConsiderations: {example['origin_considerations']}"
            answer = f"Court View: {example['considerations']}"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction, prompt, answer, task_type, jurisdiction)