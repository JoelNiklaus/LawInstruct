from datasets import load_dataset

from abstract_dataset import AbstractDataset
from abstract_dataset import JURISDICTION
from abstract_dataset import TASK_TYPE


def build_summarization_answer(input, summary):
    return f"Passage: {input}. Summary: {summary}"


class EurLexSum(AbstractDataset):

    def __init__(self):
        super().__init__(
            "EurLexSum",
            "https://huggingface.co/datasets/dennlinger/eur-lex-sum")

    def get_data(self):
        langs = {
            'bulgarian': 'bg',
            'czech': 'cs',
            'dutch': 'nl',
            'estonian': 'et',
            'french': 'fr',
            'greek': 'el',
            'irish': 'ga',
            'latvian': 'lv',
            'maltese': 'mt',
            'portuguese': 'pt',
            'slovak': 'sk',
            'spanish': 'es',
            'croatian': 'hr',
            'danish': 'da',
            'english': 'en',
            'finnish': 'fi',
            'german': 'de',
            'hungarian': 'hu',
            'italian': 'it',
            'lithuanian': 'lt',
            'polish': 'pl',
            'romanian': 'ro',
            'slovenian': 'sl',
            'swedish': 'sv'
        }
        for lang, answer_language in langs.items():
            df = load_dataset("dennlinger/eur-lex-sum", lang, split="train")

            task_type = TASK_TYPE.SUMMARIZATION
            jurisdiction = JURISDICTION.EU
            prompt_language = "en"

            instruction_bank = [
                "Summarize the following European legal document.",
                "Consider the European legal document and summarize it."
            ]
            for example in df:
                input = example["reference"]
                summary = example["summary"]
                text = f"{self.random.choice(instruction_bank)}\n\n{build_summarization_answer(input, summary)}"
                yield self.build_data_point(prompt_language, answer_language,
                                            text, task_type, jurisdiction)
