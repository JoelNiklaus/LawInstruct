from typing import Final

from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class USClassActions(AbstractDataset):
    # Legal Judgement Prediction: US Class Actions

    def __init__(self):
        super().__init__(
            "USClassActions",
            "https://huggingface.co/datasets/darrow-ai/USClassActions")

    def get_data(self):
        df = load_dataset("darrow-ai/USClassActions", split="train")
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.US
        instruction_language: Final[str] = "en"
        instruction_bank = [
            "Read the following United States class action complaint. Predict whether the complaint will be won or not. Output \"win\" or \"lose\".",
            "Will this class action complaint be successful in U.S. Court?"
        ]
        for example in df:
            instruction = self.random.choice(instruction_bank)
            prompt = {example['target_text']}
            answer = f"Likely Verdict: {example['verdict']}"
            prompt_language = "en"
            yield self.build_data_point(instruction_language, prompt_language, "en", instruction,
                                        prompt, answer, task_type, jurisdiction)
