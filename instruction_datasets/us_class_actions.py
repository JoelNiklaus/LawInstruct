from typing import Final

from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class USClassActions(AbstractDataset):
    # Legal Judgement Prediction: US Class Actions

    def __init__(self):
        super().__init__(
            "USClassActions",
            "https://huggingface.co/datasets/darrow-ai/USClassActions")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset("darrow-ai/USClassActions", split="train")
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.US
        instructions_group = "us_class_actions_win_lose"
        for example in df:
            instruction, instruction_language = instructions.sample(instructions_group)
            prompt = example['target_text']
            answer = f"Likely Verdict: {example['verdict']}"
            prompt_language = "en"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction)
