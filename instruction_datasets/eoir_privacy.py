from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class EOIRPrivacy(AbstractDataset):

    def __init__(self):
        super().__init__(
            "EOIRPrivacy",
            "https://huggingface.co/datasets/pile-of-law/eoir_privacy")

    def get_data(self):
        df = load_dataset("pile-of-law/eoir_privacy",
                          "eoir_privacy",
                          split="train")

        # TODO do we need the jurisdiction in each example of the instruction bank?
        instruction_bank = [
            "For each masked paragraph, determine if we should use a pseudonym for this case related to immigration law in the United States.",
            "Consider this paragraph from a precedential EOIR case. Should the IJ use a a pseudonym.",
            "Should the judge pseudonymize the person's name in this paragraph?"
        ]
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.US
        prompt_language = "en"

        for example in df:
            lookup = ["Don't use pseudonym.", "Use pseudonym."]
            instruction = self.random.choice(instruction_bank)
            text = f"{example['text']}\n{lookup[example['label']]}"
            yield self.build_data_point(prompt_language, "en",
                                        instruction, text, task_type,
                                        jurisdiction)
