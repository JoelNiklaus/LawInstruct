from datasets import load_dataset
from ftlangdetect import detect
from tqdm import tqdm

from abstract_dataset import AbstractDataset
from abstract_dataset import JURISDICTION
from abstract_dataset import TASK_TYPE


class XP3MT(AbstractDataset):

    def __init__(self):
        super().__init__("XP3MT",
                         "https://huggingface.co/datasets/bigscience/xP3mt")

    def get_data(self):
        jurisdiction = JURISDICTION.N_A

        # Include only code and languages where we have legal data for
        # Maybe also add 'zh', 'vi', because we have legal instruction datasets there
        _LANG = ['en', 'es', 'fr', 'pt', 'code']
        # TODO maybe treat code as a separate category so we can filter easily

        # prompts translated into other languages
        # rather use this one instead of xP3all because we compile multi_eurlex ourselves and xP3all only has two datasets more.
        # We already have enough instruction datasets and rather prioritize using more languages in the prompts
        for lang in _LANG:
            df = load_dataset("bigscience/xP3mt", lang, split="train")
            for example in tqdm(df):
                text = example["inputs"] + " " + example["targets"]
                task_type = (TASK_TYPE.CODE
                             if lang == "code" else TASK_TYPE.UNKNOWN)
                prompt_language = detect(text=example["inputs"].replace(
                    "\n", " "),
                                         low_memory=True)['lang']
                answer_language = lang if lang != "code" else "en"
                yield self.build_data_point(prompt_language, answer_language,
                                            text, task_type, jurisdiction)
