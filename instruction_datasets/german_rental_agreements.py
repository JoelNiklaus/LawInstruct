from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class GermanRentalAgreements(AbstractDataset):

    def __init__(self):
        super().__init__(
            "GermanRentalAgreements",
            "https://huggingface.co/datasets/joelito/german_rental_agreements")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset("joelito/german_rental_agreements", split="train")

        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.GERMANY
        instruction_language: str
        prompt_language = "de"

        for example in df:
            for num_classes in [3, 6, 9]:
                label = example[f"label_{num_classes}_classes"]
                sentence = example[f"text_{num_classes}_classes"]
                if sentence and label:
                    subset = "german_rental_agreements"
                    instruction, instruction_language = instructions.sample(subset)
                    # TODO: this one doesn't have any Prompt and Answer nouns...
                    prompt = sentence
                    answer = label
                    yield self.build_data_point(instruction_language,
                                                prompt_language, "de",
                                                instruction, prompt, answer,
                                                task_type, jurisdiction, subset)
