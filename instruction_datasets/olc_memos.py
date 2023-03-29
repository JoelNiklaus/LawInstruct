from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_PROMPT = ''


class OLCMemos(AbstractDataset):

    def __init__(self):
        super().__init__(
            "OLCMemos",
            "https://huggingface.co/datasets/pile-of-law/pile-of-law")

    def get_data(self):
        # OLC memos start off with a short form summary and then write the memo
        df = load_dataset("pile-of-law/pile-of-law", "olc_memos", split="train")

        instruction_bank = [
            "Write a legal research memo on the following topic.",
            "Write a memo in the style of OLC on the following legal research question.",
            "Write a memo in the form of U.S. Office of Legal Counsel.",
            "Consider the question below, write a formal legal research memo."
        ]
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.US
        instruction_language = "en"
        prompt_language = "en"

        for example in df["text"]:
            if example.startswith("b'"):
                example = example.encode().decode('unicode-escape').encode(
                    'latin1').decode('utf-8')[2:-2].strip()
            instruction = self.random.choice(instruction_bank)
            # FIXME: The memo topic is part of the text, but it may be multiline.
            #  There is no clean way to extract it beyond manual inspection.
            yield self.build_data_point(instruction_language,
                                        prompt_language,
                                        "en",
                                        instruction,
                                        _BLANK_PROMPT,
                                        example,
                                        task_type,
                                        jurisdiction,
                                        subset="olc_memos")
