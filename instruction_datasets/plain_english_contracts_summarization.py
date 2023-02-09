from datasets import load_dataset

from abstract_dataset import AbstractDataset
from abstract_dataset import JURISDICTION
from abstract_dataset import TASK_TYPE


def build_summarization_answer(input, summary):
    return f"Passage: {input}. Summary: {summary}"


def get_instruction_bank(court):
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
        task_type = TASK_TYPE.SUMMARIZATION
        jurisdiction = JURISDICTION.UNKNOWN
        prompt_language = "en"

        def get_instruction_bank(document):
            return [
                f"Summarize the following excerpt of a {document} document. ",
                f"Consider the excerpt of a {document} document and summarize it. "
            ]

        for example in df:
            instruction_bank = get_instruction_bank(example["doc"])
            input = example["original_text"]
            summary = example["reference_summary"]
            text = f"{self.random.choice(instruction_bank)}\n\n{build_summarization_answer(input, summary)}"
            yield self.build_data_point(prompt_language, "en", text, task_type,
                                        jurisdiction)
