from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class GermanRentalAgreements(AbstractDataset):

    def __init__(self):
        super().__init__(
            "GermanRentalAgreements",
            "https://huggingface.co/datasets/joelito/german_rental_agreements")

    def get_data(self):
        df = load_dataset("joelito/german_rental_agreements", split="train")

        instruction_bank = [
            "You are given a sentence from a German rental agreement. Predict the category of the sentence.",
            "Predict the category of the following sentence from a German rental agreement.",
        ]
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.GERMANY
        prompt_language = "en"

        for example in df:
            for num_classes in [3, 6, 9]:
                label = example[f"label_{num_classes}_classes"]
                sentence = example[f"text_{num_classes}_classes"]
                if sentence and label:
                    text = f"{self.random.choice(instruction_bank)}\n\n{sentence}\n{label}"
                    yield self.build_data_point(prompt_language, "de", text,
                                                task_type, jurisdiction)
