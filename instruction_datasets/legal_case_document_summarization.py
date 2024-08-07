import string

from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


def build_summarization_answer(input: str, summary: str) -> tuple[str, str]:
    prompt = f"Passage: {input}"
    # Attach a period if the prompt doesn't end in punctuation.
    if prompt[-1] not in string.punctuation:
        prompt += "."

    answer = f"Summary: {summary}"

    return prompt, answer


def get_instruction_bank(court: str) -> list[str]:
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

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset("joelito/legal_case_document_summarization",
                          split="train")
        task_type = TaskType.SUMMARIZATION
        instruction_language = "en"
        prompt_language = "en"
        answer_language = "en"

        subset = "legal_case_summarization_india"
        indian_df = df.filter(lambda example: "IN" in example["dataset_name"])
        for example in indian_df:
            instruction_bank = get_instruction_bank("Indian Supreme Court case")
            jurisdiction = Jurisdiction.INDIA

            input = example["judgement"]
            summary = example["summary"]
            instruction, instruction_language = instructions.sample(subset)
            prompt, answer = build_summarization_answer(input, summary)
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, answer,
                                        task_type, jurisdiction, subset)

        subset = "legal_case_summarization_uk"
        uk_df = df.filter(lambda example: "UK" in example["dataset_name"])
        for example in uk_df:
            instruction_bank = get_instruction_bank("U.K. Supreme Court case")
            jurisdiction = Jurisdiction.UK

            input = example["judgement"]
            summary = example["summary"]
            instruction, instruction_language = instructions.sample(subset)
            prompt, answer = build_summarization_answer(input, summary)
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, answer,
                                        task_type, jurisdiction, subset)
