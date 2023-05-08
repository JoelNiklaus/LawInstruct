import string

from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


def build_summarization_answer(input: str, summary: str) -> tuple[str, str]:
    prompt = f"Passage: {input}"
    if prompt[-1] not in string.punctuation:
        prompt += "."

    answer = f"Summary: {summary}"

    return prompt, answer


class EurLexSum(AbstractDataset):

    def __init__(self):
        super().__init__(
            "EurLexSum",
            "https://huggingface.co/datasets/dennlinger/eur-lex-sum")

    def get_data(self, instructions: instruction_manager.InstructionManager):
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

            task_type = TaskType.SUMMARIZATION
            jurisdiction = Jurisdiction.EU
            instruction_language: str
            prompt_language = "en"

            for example in df:
                input = example["reference"]
                summary = example["summary"]
                instruction, instruction_language = instructions.sample(
                    "eur_lex_sum")
                prompt, answer = build_summarization_answer(input, summary)
                yield self.build_data_point(instruction_language,
                                            prompt_language, answer_language,
                                            instruction, prompt, answer,
                                            task_type, jurisdiction)
