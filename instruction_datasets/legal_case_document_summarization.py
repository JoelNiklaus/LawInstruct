from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


def build_summarization_answer(input, summary):
    return f"Passage: {input}. Summary: {summary}"


def get_instruction_bank(court):
    return [
        f"Summarize the document of the {court}. ",
        f"Consider the document of the {court} and summarize it. "
    ]


class LegalCaseDocumentSummarization(AbstractDataset):

    def __init__(self):
        super().__init__(
            "LegalCaseDocumentSummarization",
            "https://huggingface.co/datasets/joelito/legal_case_document_summarization"
        )

    def get_data(self):
        df = load_dataset("joelito/legal_case_document_summarization",
                          split="train")
        task_type = TaskType.SUMMARIZATION
        prompt_language = "en"

        for example in df:
            if "IN" in example["dataset_name"]:
                instruction_bank = get_instruction_bank(
                    "Indian Supreme Court case")
                jurisdiction = Jurisdiction.INDIA
            elif "UK" in example["dataset_name"]:
                instruction_bank = get_instruction_bank(
                    "U.K. Supreme Court case")
                jurisdiction = Jurisdiction.UK
            else:
                continue
            input = example["judgement"]
            summary = example["summary"]
            instruction = self.random.choice(instruction_bank)
            text = build_summarization_answer(input, summary)
            yield self.build_data_point(prompt_language, "en", instruction,
                                        text, task_type, jurisdiction)
