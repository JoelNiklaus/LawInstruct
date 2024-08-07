from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

ner_fine_tags = [
    'B-AN', 'B-EUN', 'B-GRT', 'B-GS', 'B-INN', 'B-LD', 'B-LDS', 'B-LIT',
    'B-MRK', 'B-ORG', 'B-PER', 'B-RR', 'B-RS', 'B-ST', 'B-STR', 'B-UN', 'B-VO',
    'B-VS', 'B-VT', 'I-AN', 'I-EUN', 'I-GRT', 'I-GS', 'I-INN', 'I-LD', 'I-LDS',
    'I-LIT', 'I-MRK', 'I-ORG', 'I-PER', 'I-RR', 'I-RS', 'I-ST', 'I-STR', 'I-UN',
    'I-VO', 'I-VS', 'I-VT', 'O'
]
ner_coarse_tags = [
    'B-LIT', 'B-LOC', 'B-NRM', 'B-ORG', 'B-PER', 'B-REG', 'B-RS', 'I-LIT',
    'I-LOC', 'I-NRM', 'I-ORG', 'I-PER', 'I-REG', 'I-RS', 'O'
]

NER_DELIMITER = "|"


def get_ner_instruction(ner_tags: list[str]) -> str:
    return (f"Predict the named entity types for"
            f" each token (delimited by '{NER_DELIMITER}')."
            f" The named entities are: {' '.join(ner_tags)}.")


def build_ner_answer(tokens: list[str], tags: list[str]) -> tuple[str, str]:
    return (f"Sentence: {NER_DELIMITER.join(tokens)}",
            f"Named Entity Types: {NER_DELIMITER.join(tags)}")


class GermanLER(AbstractDataset):

    def __init__(self):
        super().__init__(
            "GermanLER",
            "https://huggingface.co/datasets/elenanereiss/german-ler")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset("elenanereiss/german-ler", split="train")
        task_type = TaskType.NAMED_ENTITY_RECOGNITION
        jurisdiction = Jurisdiction.GERMANY
        instruction_language = "en"
        prompt_language = "de"
        answer_language = "ner"

        introduction_sentence = "Consider the following sentence from a German federal court decision."
        instruction_bank_fine = [
            f"{introduction_sentence} {get_ner_instruction(ner_fine_tags)}",
        ]
        instruction_bank_coarse = [
            f"{introduction_sentence} {get_ner_instruction(ner_coarse_tags)}"
        ]
        for example in df:
            tags = [ner_fine_tags[tag] for tag in example["ner_tags"]]
            instruction = self.random.choice(instruction_bank_fine)
            prompt, answer = build_ner_answer(example['tokens'], tags)
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, "fine")

        for example in df:
            tags = [ner_coarse_tags[tag] for tag in example["ner_coarse_tags"]]
            instruction = self.random.choice(instruction_bank_coarse)
            prompt, answer = build_ner_answer(example['tokens'], tags)
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, "coarse")
