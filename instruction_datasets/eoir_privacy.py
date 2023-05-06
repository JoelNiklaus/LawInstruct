from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class EOIRPrivacy(AbstractDataset):

    def __init__(self):
        super().__init__(
            "EOIRPrivacy",
            "https://huggingface.co/datasets/pile-of-law/eoir_privacy")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset("pile-of-law/eoir_privacy",
                          "eoir_privacy",
                          split="train")

        # TODO do we need the jurisdiction in each example of the instruction bank?
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"

        for example in df:
            lookup = ["Don't use pseudonym.", "Use pseudonym."]
            instruction, instruction_language = instructions.sample('eoir_privacy')
            prompt = example['text']
            answer = lookup[example['label']]
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction)
