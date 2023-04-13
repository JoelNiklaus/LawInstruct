from datasets import load_dataset
from ftlangdetect import detect
from tqdm import tqdm

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''
_BLANK_INSTRUCTION_LANGUAGE = 'zxx'


class XP3MT(AbstractDataset):

    def __init__(self):
        super().__init__("XP3MT",
                         "https://huggingface.co/datasets/bigscience/xP3mt")

    def get_data(self):
        jurisdiction = Jurisdiction.N_A

        # Include only code and languages where we have legal data for
        # Maybe also add 'zh', 'vi', because we have legal instruction datasets there
        _LANG = ['en', 'es', 'fr', 'pt', 'code']
        # TODO maybe treat code as a separate category, so we can filter easily

        # prompts translated into other languages
        # rather use this one instead of xP3all because we compile multi_eurlex ourselves and xP3all only has two datasets more.
        # We already have enough instruction datasets and rather prioritize using more languages in the prompts
        for lang in _LANG:
            df = load_dataset("bigscience/xP3mt", lang, split="train")
            for example in tqdm(df):
                prompt = example["inputs"]
                answer = example["targets"]
                task_type = (TaskType.CODE
                             if lang == "code" else TaskType.UNKNOWN)
                prompt_language = detect(text=example["inputs"].replace(
                    "\n", " "),
                                         low_memory=True)['lang']
                answer_language = lang if lang != "code" else "en"
                yield self.build_data_point(_BLANK_INSTRUCTION_LANGUAGE,
                                            prompt_language, answer_language,
                                            _BLANK_INSTRUCTION, prompt, answer,
                                            task_type, jurisdiction)
