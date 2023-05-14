from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

_BLANK_PROMPT = ''


class OLCMemos(AbstractDataset):

    def __init__(self):
        super().__init__(
            "OLCMemos",
            "https://huggingface.co/datasets/pile-of-law/pile-of-law")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        # OLC memos start off with a short form summary and then write the memo
        subset = "olc_memos"
        df = load_dataset("pile-of-law/pile-of-law", subset, split="train")

        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"

        for example in df["text"]:
            if example.startswith("b'"):
                example = example.encode().decode('unicode-escape').encode(
                    'latin1').decode('utf-8')[2:-2].strip()
            instruction, instruction_language = instructions.sample(subset)
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
                                        subset=subset)
