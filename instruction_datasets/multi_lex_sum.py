from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


def build_summarization_answer(input, summary):
    return f"Passage: {input}. Summary: {summary}"


class MultiLexSum(AbstractDataset):

    def __init__(self):
        super().__init__(
            "MultiLexSum",
            "https://huggingface.co/datasets/allenai/multi_lexsum")

    def get_data(self):
        df = load_dataset("allenai/multi_lexsum", split="train")
        task_type = TaskType.SUMMARIZATION
        jurisdiction = Jurisdiction.US
        prompt_language = "en"

        instruction_bank = [
            "Summarize the following summary of a US legal document further. ",
            "Consider the summary of a US legal document and summarize it further. "
        ]
        for example in df:
            input = example["summary/long"]
            if example["summary/short"]:
                summary = example["summary/short"]
                instruction = self.random.choice(instruction_bank)
                text = build_summarization_answer(input, summary)
                yield self.build_data_point(prompt_language, "en", instruction,
                                            text, task_type, jurisdiction)
            if example["summary/tiny"]:
                summary = example["summary/tiny"]
                instruction = self.random.choice(instruction_bank)
                text = build_summarization_answer(input, summary)
                yield self.build_data_point(prompt_language, "en", instruction,
                                            text, task_type, jurisdiction)
            if example["summary/short"] and example["summary/tiny"]:
                input = example["summary/short"]
                summary = example["summary/tiny"]
                instruction = self.random.choice(instruction_bank)
                text = build_summarization_answer(input, summary)
                yield self.build_data_point(prompt_language, "en", instruction,
                                            text, task_type, jurisdiction)
