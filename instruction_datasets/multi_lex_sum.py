import string

from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


def build_summarization_answer(input: str, summary: str) -> tuple[str, str]:
    prompt = f"Passage: {input}"
    # Add a period if the last character is not a punctuation.
    if prompt[-1] not in string.punctuation:
        prompt += "."

    answer = f"Summary: {summary}"
    return prompt, answer


class MultiLexSum(AbstractDataset):

    def __init__(self):
        super().__init__(
            "MultiLexSum",
            "https://huggingface.co/datasets/allenai/multi_lexsum")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset("allenai/multi_lexsum", "v20230518", split="train")
        task_type = TaskType.SUMMARIZATION
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"

        for example in df:
            input = example["summary/long"]
            subset = "multi_lex_sum"
            if example["summary/short"]:
                summary = example["summary/short"]
                instruction, instruction_language = instructions.sample(subset)
                prompt, answer = build_summarization_answer(input, summary)
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, "long_to_short")
            if example["summary/tiny"]:
                summary = example["summary/tiny"]
                instruction, instruction_language = instructions.sample(subset)
                prompt, answer = build_summarization_answer(input, summary)
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, "long_to_tiny")
            if example["summary/short"] and example["summary/tiny"]:
                input = example["summary/short"]
                summary = example["summary/tiny"]
                instruction, instruction_language = instructions.sample(subset)
                prompt, answer = build_summarization_answer(input, summary)
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, "short_to_tiny")
