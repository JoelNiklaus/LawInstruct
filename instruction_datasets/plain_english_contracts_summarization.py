import string

from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


def build_summarization_answer(input: str, summary: str) -> tuple[str, str]:
    prompt = f"Passage: {input}"
    # Add a period if the last character is not a punctuation.
    if prompt[-1] not in string.punctuation:
        prompt += "."

    answer = f"Summary: {summary}"
    return prompt, answer


def get_instruction_bank(court: str) -> list[str]:
    return [
        f"Summarize the document of the {court}. ",
        f"Consider the document of the {court} and summarize it. "
    ]


class PlainEnglishContractsSummarization(AbstractDataset):

    def __init__(self):
        super().__init__(
            "PlainEnglishContractsSummarization",
            "https://huggingface.co/datasets/joelito/plain_english_contracts_summarization"
        )

    def get_data(self):
        df = load_dataset("joelito/plain_english_contracts_summarization",
                          split="train")
        task_type = TaskType.SUMMARIZATION
        jurisdiction = Jurisdiction.UNKNOWN
        instruction_language = "en"
        prompt_language = "en"

        def get_instruction_bank(document):
            return [
                f"Summarize the following excerpt of a {document} document. ",
                f"Consider the excerpt of a {document} document and summarize it. "
            ]

        for example in df:
            instruction_bank = get_instruction_bank(example["doc"])
            input_ = example["original_text"]
            summary = example["reference_summary"]
            instruction = self.random.choice(instruction_bank)
            prompt, answer = build_summarization_answer(input_, summary)
            yield self.build_data_point(instruction_language, prompt_language, "en", instruction,
                                        prompt, answer, task_type, jurisdiction)
