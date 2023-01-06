from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE

ner_fine_tags = ['B-AN', 'B-EUN', 'B-GRT', 'B-GS', 'B-INN', 'B-LD', 'B-LDS', 'B-LIT', 'B-MRK', 'B-ORG', 'B-PER',
                 'B-RR',
                 'B-RS', 'B-ST', 'B-STR', 'B-UN', 'B-VO', 'B-VS', 'B-VT', 'I-AN', 'I-EUN', 'I-GRT', 'I-GS',
                 'I-INN',
                 'I-LD', 'I-LDS', 'I-LIT', 'I-MRK', 'I-ORG', 'I-PER', 'I-RR', 'I-RS', 'I-ST', 'I-STR', 'I-UN',
                 'I-VO',
                 'I-VS', 'I-VT', 'O']
ner_coarse_tags = ['B-LIT', 'B-LOC', 'B-NRM', 'B-ORG', 'B-PER', 'B-REG', 'B-RS', 'I-LIT', 'I-LOC', 'I-NRM',
                   'I-ORG',
                   'I-PER', 'I-REG', 'I-RS', 'O']

NER_DELIMITER = "|"

def get_ner_instruction(ner_tags):
    return f"Predict the named entity types for each token (delimited by '{NER_DELIMITER}'). " \
           f"The named entities are: {' '.join(ner_tags)}."


def build_ner_answer(tokens, tags):
    f"Sentence: {NER_DELIMITER.join(tokens)}\n\n" \
    f"Named Entity Types: {NER_DELIMITER.join(tags)}\n\n"


class GermanLER(AbstractDataset):
    def __init__(self):
        super().__init__("GermanLER", "https://huggingface.co/datasets/elenanereiss/german-ler")

    def get_data(self):
        df = load_dataset("elenanereiss/german-ler", split="train")
        task_type = TASK_TYPE.NAMED_ENTITY_RECOGNITION
        jurisdiction = JURISDICTION.GERMANY
        prompt_language = "en"
        answer_language = "de"

        introduction_sentence = "Consider the following sentence from a German federal court decision."
        instruction_bank_fine = [f"{introduction_sentence} {get_ner_instruction(ner_fine_tags)}", ]
        instruction_bank_coarse = [f"{introduction_sentence} {get_ner_instruction(ner_coarse_tags)}"]
        for example in df:
            text = f"{self.random.choice(instruction_bank_fine)}\n\n{build_ner_answer(example['tokens'], example['ner_tags'])}"
            yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)

            text = f"{self.random.choice(instruction_bank_coarse)}\n\n{build_ner_answer(example['tokens'], example['ner_coarse_tags'])}"
            yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)
