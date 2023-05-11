import ast
from collections.abc import Collection
from collections.abc import Sequence
from typing import Final

from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

ner_class_mapping = {
    "lener_br": [
        "O",
        "B-ORGANIZACAO",
        "I-ORGANIZACAO",
        "B-PESSOA",
        "I-PESSOA",
        "B-TEMPO",
        "I-TEMPO",
        "B-LOCAL",
        "I-LOCAL",
        "B-LEGISLACAO",
        "I-LEGISLACAO",
        "B-JURISPRUDENCIA",
        "I-JURISPRUDENCIA",
    ],
    "legalnero": [
        'O',
        'B-TIME',
        'I-TIME',
        'B-LEGAL',
        'I-LEGAL',
        'B-ORG',
        'I-ORG',
        'B-LOC',
        'I-LOC',
        'B-PER',
        'I-PER',
    ],
    "greek_legal_ner": [
        'O', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LEG-REFS', 'I-LEG-REFS',
        'B-PUBLIC-DOCS', 'I-PUBLIC-DOCS', 'B-PERSON', 'I-PERSON', 'B-FACILITY',
        'I-FACILITY', 'B-LOCATION-UNK', 'I-LOCATION-UNK', 'B-LOCATION-NAT',
        'I-LOCATION-NAT'
    ],
    "mapa_coarse": [
        'O', 'B-ORGANISATION', 'I-ORGANISATION', 'B-ADDRESS', 'I-ADDRESS',
        'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON', 'B-AMOUNT', 'I-AMOUNT',
        'B-TIME', 'I-TIME'
    ],
    "mapa_fine": [
        'O',
        'B-BUILDING',
        'I-BUILDING',
        'B-CITY',
        'I-CITY',
        'B-COUNTRY',
        'I-COUNTRY',
        'B-PLACE',
        'I-PLACE',
        'B-TERRITORY',
        'I-TERRITORY',
        'I-UNIT',
        'B-UNIT',
        'B-VALUE',
        'I-VALUE',
        'B-YEAR',
        'I-YEAR',
        'B-STANDARD ABBREVIATION',
        'I-STANDARD ABBREVIATION',
        'B-MONTH',
        'I-MONTH',
        'B-DAY',
        'I-DAY',
        'B-AGE',
        'I-AGE',
        'B-ETHNIC CATEGORY',
        'I-ETHNIC CATEGORY',
        'B-FAMILY NAME',
        'I-FAMILY NAME',
        'B-INITIAL NAME',
        'I-INITIAL NAME',
        'B-MARITAL STATUS',
        'I-MARITAL STATUS',
        'B-PROFESSION',
        'I-PROFESSION',
        'B-ROLE',
        'I-ROLE',
        'B-NATIONALITY',
        'I-NATIONALITY',
        'B-TITLE',
        'I-TITLE',
        'B-URL',
        'I-URL',
        'B-TYPE',
        'I-TYPE',
    ],
}

INSTRUCTION_GROUPS: Final[tuple[str, ...]] = ('brazilian_court_decisions_judgment', 'brazilian_court_decisions_unanimity', 'german_argument_mining', 'greek_legal_code_chapter', 'greek_legal_code_subject', 'greek_legal_code_volume', 'swiss_judgment_prediction', 'online_terms_of_service_unfairness_levels', 'online_terms_of_service_clause_topics', 'covid19_emergency_event', 'multi_eurlex_level_1', 'multi_eurlex_level_2', 'multi_eurlex_level_3', 'greek_legal_ner', 'legalnero', 'lener_br', 'mapa_coarse', 'mapa_fine')

TASK_CODE_MAPPING = {
    'brazilian_court_decisions_judgment': 'SLTC',
    'brazilian_court_decisions_unanimity': 'SLTC',
    'swiss_judgment_prediction': 'SLTC',
    'german_argument_mining': 'SLTC',
    'greek_legal_code_chapter': 'SLTC',
    'greek_legal_code_subject': 'SLTC',
    'greek_legal_code_volume': 'SLTC',
    'online_terms_of_service_unfairness_levels': 'SLTC',
    'online_terms_of_service_clause_topics': 'MLTC',
    'covid19_emergency_event': 'MLTC',
    'multi_eurlex_level_1': 'MLTC',
    'multi_eurlex_level_2': 'MLTC',
    'multi_eurlex_level_3': 'MLTC',
    'greek_legal_ner': 'NER',
    'legalnero': 'NER',
    'lener_br': 'NER',
    'mapa_coarse': 'NER',
    'mapa_fine': 'NER',
}

JURISDICTION_MAPPING = {
    'brazilian_court_decisions_judgment': Jurisdiction.BRAZIL,
    'brazilian_court_decisions_unanimity': Jurisdiction.BRAZIL,
    'swiss_judgment_prediction': Jurisdiction.SWITZERLAND,
    'german_argument_mining': Jurisdiction.GERMANY,
    'greek_legal_code_chapter': Jurisdiction.GREECE,
    'greek_legal_code_subject': Jurisdiction.GREECE,
    'greek_legal_code_volume': Jurisdiction.GREECE,
    'online_terms_of_service_unfairness_levels': Jurisdiction.UNKNOWN,
    'online_terms_of_service_clause_topics': Jurisdiction.UNKNOWN,
    'covid19_emergency_event': Jurisdiction.UNKNOWN,
    'multi_eurlex_level_1': Jurisdiction.EU,
    'multi_eurlex_level_2': Jurisdiction.EU,
    'multi_eurlex_level_3': Jurisdiction.EU,
    'greek_legal_ner': Jurisdiction.GREECE,
    'legalnero': Jurisdiction.ROMANIA,
    'lener_br': Jurisdiction.BRAZIL,
    'mapa_coarse': Jurisdiction.EU,
    'mapa_fine': Jurisdiction.EU,
}

NER_DELIMITER = "|"


def get_ner_instruction(ner_tags: Collection[str]) -> str:
    return f"Predict the named entity types for each token (delimited by '{NER_DELIMITER}'). " \
           f"The named entities are: {' '.join(ner_tags)}."


def build_ner_answer(tokens: Sequence[str],
                     tags: Collection[str]) -> tuple[str, str]:
    prompt = f"Sentence: {NER_DELIMITER.join(tokens)}"
    answer = f"Named Entity Types: {NER_DELIMITER.join(tags)}"
    return prompt, answer


class LEXTREME(AbstractDataset):
    # swiss_judgment_prediction is handled separately

    def __init__(self):
        super().__init__("LEXTREME",
                         "https://huggingface.co/datasets/joelito/lextreme")

    def get_data(self, instructions_: instruction_manager.InstructionManager):
        instruction_language: str
        for subset in INSTRUCTION_GROUPS:
            dataset = load_dataset("joelito/lextreme", subset, split="train")
            jurisdiction = JURISDICTION_MAPPING[subset]
            task_code = TASK_CODE_MAPPING[subset]

            if task_code == 'SLTC':
                class_label = dataset.features["label"]
            elif task_code == 'NER':
                label_classes = ner_class_mapping[subset]

            for example in dataset:
                # get correct labels
                correct_labels: list[str]
                if task_code == 'SLTC':
                    correct_label = class_label.int2str(
                        example['label'])  # get label name for correct label
                    correct_labels = correct_label if isinstance(
                        correct_label, list) else [correct_label]
                elif task_code == 'MLTC':
                    correct_labels = list(
                        map(str, example['label']
                           ))  # here we don't have any mapping to label names
                elif task_code == 'NER':
                    correct_labels = [
                        label_classes[label] for label in example['label']
                    ]

                if subset in ['online_terms_of_service_clause_topics', 'covid19_emergency_event']:
                    correct_labels = [chr(int(num) + 65) for num in correct_labels]  # convert to letters

                answers: list[tuple[str, str, str]]
                if task_code in ['SLTC', 'MLTC']:
                    input_text = example['input']
                    if subset.startswith('multi_eurlex'):
                        input_text = ast.literal_eval(input_text)
                        assert isinstance(input_text, dict)
                        answers = [(f"Passage {input_text[lang]}",
                                    f"Labels: {','.join(correct_labels)}", lang)
                                   for lang, text in input_text.items()]
                    else:
                        answers = [(f"Passage {input_text}",
                                    f"Labels: {','.join(correct_labels)}",
                                    example['language'])]

                elif task_code == 'NER':
                    prompt, answer = build_ner_answer(example["input"],
                                                      correct_labels)
                    answers = [(prompt, answer, example['language'])]

                for prompt, answer, lang in answers:
                    task_type = TaskType.NAMED_ENTITY_RECOGNITION if task_code == 'NER' else TaskType.TEXT_CLASSIFICATION
                    prompt_language = "en"
                    instruction, instruction_language = instructions_.sample(subset)
                    yield self.build_data_point(instruction_language,
                                                prompt_language,
                                                lang,
                                                instruction,
                                                prompt,
                                                answer,
                                                task_type,
                                                jurisdiction,
                                                subset=subset)
