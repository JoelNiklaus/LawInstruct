import ast

from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE

ner_class_mapping = {
    "lener_br": [
        "O", "B-ORGANIZACAO", "I-ORGANIZACAO", "B-PESSOA", "I-PESSOA", "B-TEMPO", "I-TEMPO", "B-LOCAL", "I-LOCAL",
        "B-LEGISLACAO", "I-LEGISLACAO", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA",
    ],
    "legalnero": [
        'O', 'B-TIME', 'I-TIME', 'B-LEGAL', 'I-LEGAL', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER',
    ],
    "greek_legal_ner": [
        'O', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LEG-REFS', 'I-LEG-REFS', 'B-PUBLIC-DOCS', 'I-PUBLIC-DOCS',
        'B-PERSON', 'I-PERSON', 'B-FACILITY', 'I-FACILITY', 'B-LOCATION-UNK', 'I-LOCATION-UNK', 'B-LOCATION-NAT',
        'I-LOCATION-NAT'
    ],
    "mapa_coarse": [
        'O', 'B-ORGANISATION', 'I-ORGANISATION', 'B-ADDRESS', 'I-ADDRESS', 'B-DATE', 'I-DATE', 'B-PERSON',
        'I-PERSON',
        'B-AMOUNT', 'I-AMOUNT', 'B-TIME', 'I-TIME'
    ],
    "mapa_fine": [
        'O', 'B-BUILDING', 'I-BUILDING', 'B-CITY', 'I-CITY', 'B-COUNTRY', 'I-COUNTRY', 'B-PLACE', 'I-PLACE',
        'B-TERRITORY', 'I-TERRITORY', 'I-UNIT', 'B-UNIT', 'B-VALUE', 'I-VALUE', 'B-YEAR', 'I-YEAR',
        'B-STANDARD ABBREVIATION', 'I-STANDARD ABBREVIATION', 'B-MONTH', 'I-MONTH', 'B-DAY', 'I-DAY', 'B-AGE',
        'I-AGE',
        'B-ETHNIC CATEGORY', 'I-ETHNIC CATEGORY', 'B-FAMILY NAME', 'I-FAMILY NAME', 'B-INITIAL NAME',
        'I-INITIAL NAME',
        'B-MARITAL STATUS', 'I-MARITAL STATUS', 'B-PROFESSION', 'I-PROFESSION', 'B-ROLE', 'I-ROLE', 'B-NATIONALITY',
        'I-NATIONALITY', 'B-TITLE', 'I-TITLE', 'B-URL', 'I-URL', 'B-TYPE', 'I-TYPE',
    ],
}

instructions_for_subsets = {
    "brazilian_court_decisions_judgment": "In this task, you are given the case description from a decision heard at the State Supreme Court of Alagoas (Brazil). "
                                          "Predict the judgment of the case "
                                          "(no: The appeal was denied, "
                                          "partial: For partially favourable decisions, "
                                          "yes: For fully favourable decisions)",
    "brazilian_court_decisions_unanimity": "In this task, you are given the case description from a decision heard at the State Supreme Court of Alagoas (Brazil). "
                                           "Predict the unanimity of the case (unanimity, not-unanimity, not_determined)",
    "german_argument_mining": "In this task, you are given sentences from German court decisions. "
                              "Predict the major component of German Urteilsstil "
                              "(conclusion: Overall result, "
                              "definition: Abstract legal facts and consequences, "
                              "subsumption: Determination sentence / Concrete facts, "
                              "other: Anything else)",
    "greek_legal_code_chapter": "In this task, you are given a Greek legislative document. "
                                "Predict the chapter level category of the 'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    "greek_legal_code_subject": "In this task, you are given a Greek legislative document. "
                                "Predict the subject level category of the 'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    "greek_legal_code_volume": "In this task, you are given a Greek legislative document. "
                               "Predict the volume level category of the 'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    "online_terms_of_service_unfairness_levels": "In this task, you are given a sentence from a Terms of Service (ToS) document. "
                                                 "Predict the unfairness level of the sentence (potentially_unfair, clearly_unfair, clearly_fair, untagged)",
    "online_terms_of_service_clause_topics": "In this task, you are given a sentence from a Terms of Service (ToS) document. "
                                             "Predict the clause topics of the sentence "
                                             "(0: Arbitration, "
                                             "1: Unilateral change, "
                                             "2: Content removal, "
                                             "3: Jurisdiction, "
                                             "4: Choice of law, "
                                             "5: Limitation of liability, "
                                             "6: Unilateral termination, "
                                             "7: Contract by using, "
                                             "8: Privacy included)",
    "covid19_emergency_event": "In this task, you are given a sentence from a European legislative document. "
                               "Predict the applicable measurements against COVID-19 "
                               "(0: State of Emergency, "
                               "1: Restrictions of fundamental rights and civil liberties, "
                               "2: Restrictions of daily liberties, "
                               "3: Closures / lockdown, "
                               "4: Suspension of international cooperation and commitments, "
                               "5: Police mobilization, "
                               "6: Army mobilization, "
                               "7: Government oversight)",
    "multi_eurlex_level_1": "In this task, you are given a document from an EU law. "
                            "Predict the level 1 concept in the EUROVOC taxonomy.",
    "multi_eurlex_level_2": "In this task, you are given a document from an EU law. "
                            "Predict the level 2 concept in the EUROVOC taxonomy.",
    "multi_eurlex_level_3": "In this task, you are given a document from an EU law. "
                            "Predict the level 3 concept in the EUROVOC taxonomy.",
    "greek_legal_ner": "In this task, you are given a sentence from Greek legislation. "
                       "Predict the named entity type for each token.",
    "legalnero": "In this task, you are given a sentence from Romanian legislation. "
                 "Predict the named entity type for each token.",
    "lener_br": "In this task, you are given a sentence from Brazilian legal documents (court decisions and legislation). "
                "Predict the named entity type for each token.",
    "mapa_coarse": "In this task, you are given a sentence from the EUR-Lex database. "
                   "Predict the coarse grained named entity type for each token.",
    "mapa_fine": "In this task, you are given a sentence from the EUR-Lex database. "
                 "Predict the fine grained named entity type for each token.",
}

TASK_CODE_MAPPING = {
    'brazilian_court_decisions_judgment': 'SLTC',
    'brazilian_court_decisions_unanimity': 'SLTC',
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
    'brazilian_court_decisions_judgment': JURISDICTION.BRAZIL,
    'brazilian_court_decisions_unanimity': JURISDICTION.BRAZIL,
    'german_argument_mining': JURISDICTION.GERMANY,
    'greek_legal_code_chapter': JURISDICTION.GREECE,
    'greek_legal_code_subject': JURISDICTION.GREECE,
    'greek_legal_code_volume': JURISDICTION.GREECE,
    'online_terms_of_service_unfairness_levels': JURISDICTION.UNKNOWN,
    'online_terms_of_service_clause_topics': JURISDICTION.UNKNOWN,
    'covid19_emergency_event': JURISDICTION.UNKNOWN,
    'multi_eurlex_level_1': JURISDICTION.EU,
    'multi_eurlex_level_2': JURISDICTION.EU,
    'multi_eurlex_level_3': JURISDICTION.EU,
    'greek_legal_ner': JURISDICTION.GREECE,
    'legalnero': JURISDICTION.ROMANIA,
    'lener_br': JURISDICTION.BRAZIL,
    'mapa_coarse': JURISDICTION.EU,
    'mapa_fine': JURISDICTION.EU,
}

NER_DELIMITER = "|"


def get_ner_instruction(ner_tags):
    return f"Predict the named entity types for each token (delimited by '{NER_DELIMITER}'). " \
           f"The named entities are: {' '.join(ner_tags)}."


def build_ner_answer(tokens, tags):
    f"Sentence: {NER_DELIMITER.join(tokens)}\n\n" \
    f"Named Entity Types: {NER_DELIMITER.join(tags)}\n\n"


class LEXTREME(AbstractDataset):
    # swiss_judgment_prediction is handled separately

    def __init__(self):
        super().__init__("LEXTREME", "https://huggingface.co/datasets/joelito/lextreme")

    def get_data(self):
        for subset, instructions in instructions_for_subsets.items():
            dataset = load_dataset("joelito/lextreme", subset, split="train")
            jurisdiction = JURISDICTION_MAPPING[subset]
            task_code = TASK_CODE_MAPPING[subset]
            if task_code == "NER":
                instructions += " " + get_ner_instruction(ner_class_mapping[subset])

            if task_code == 'SLTC':
                class_label = dataset.features["label"]
            elif task_code == 'NER':
                label_classes = ner_class_mapping[subset]

            for example in dataset:
                # get correct labels
                if task_code == 'SLTC':
                    correct_label = class_label.int2str(example['label'])  # get label name for correct label
                    correct_labels = correct_label if isinstance(correct_label, list) else [correct_label]
                elif task_code == 'MLTC':
                    correct_labels = list(map(str, example['label']))  # here we don't have any mapping to label names
                elif task_code == 'NER':
                    correct_labels = [label_classes[label] for label in example['label']]

                if task_code in ['SLTC', 'MLTC']:
                    input_text = example['input']
                    if 'multi_eurlex' in subset:
                        input_text = ast.literal_eval(input_text)
                        assert isinstance(input_text, dict)
                        answers = [(
                            f"Passage {input_text[lang]} Labels: {','.join(correct_labels)}",
                            lang) for lang, text in input_text.items()]
                    else:
                        answers = [(f"Passage {input_text} Labels: {','.join(correct_labels)}", example['language'])]

                elif task_code == 'NER':
                    answers = [(build_ner_answer(example["input"], correct_labels), example['language'])]

                for answer, lang in answers:
                    text = f"{instructions}\n\n{answer}"
                    task_type = TASK_TYPE.NAMED_ENTITY_RECOGNITION if task_code == 'NER' else TASK_TYPE.TEXT_CLASSIFICATION
                    prompt_language = "en"
                    yield self.build_data_point(prompt_language, lang, text, task_type, jurisdiction, subset)
