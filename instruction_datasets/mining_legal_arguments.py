from datasets import load_dataset

from abstract_dataset import AbstractDataset
from abstract_dataset import JURISDICTION
from abstract_dataset import TASK_TYPE

NER_DELIMITER = "|"


def get_ner_instruction(ner_tags):
    return f"Predict the named entity types for each token (delimited by '{NER_DELIMITER}'). " \
           f"The named entities are: {' '.join(ner_tags)}."


def build_ner_answer(tokens, tags):
    return f"Sentence: {NER_DELIMITER.join(tokens)}\n\n" \
           f"Named Entity Types: {NER_DELIMITER.join(tags)}\n\n"


def get_all_ner_labels(df, labels_column_name="labels"):
    all_labels = set()
    for example in df:
        all_labels.update(example[labels_column_name])
    return all_labels


class MiningLegalArguments(AbstractDataset):

    def __init__(self):
        super().__init__("MiningLegalArguments",
                         "https://github.com/trusthlt/mining-legal-arguments")

    def get_data(self):
        task_type = TASK_TYPE.NAMED_ENTITY_RECOGNITION
        jurisdiction = JURISDICTION.EU
        prompt_language = "en"
        for type in ["agent", "argType"]:
            source = f"https://huggingface.co/datasets/joelito/mining_legal_arguments_{type}"
            df = load_dataset(f"joelito/mining_legal_arguments_{type}",
                              split="train")
            all_labels = get_all_ner_labels(df)
            introduction_sentence = "Consider the following sentence from an ECtHR decision. "
            instruction_bank = [
                f"{introduction_sentence} {get_ner_instruction(all_labels)}",
            ]
            for example in df:
                text = f"{self.random.choice(instruction_bank)}\n\n{build_ner_answer(example['tokens'], example['labels'])}"
                yield self.build_data_point(prompt_language,
                                            "en",
                                            text,
                                            task_type,
                                            jurisdiction,
                                            subset=type)
