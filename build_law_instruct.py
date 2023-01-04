import ast

from datasets import load_dataset
from collections import defaultdict
from lxml import etree

import os
import random
from typing import List
from tqdm import tqdm
import json
from bs4 import BeautifulSoup
import pandas as pd

from utils import write_json_line, MAX_FILE_SIZE, get_output_file_name

import glob

try:
    import lzma as xz
except ImportError:
    import pylzma as xz
import toml


# To be reconsidered later
# LegalSum (https://github.com/sebimo/LegalSum) ==> complicated to read because of norms and would require large preprocessing. Additionally, contains very long sequences. leave out for the moment
# LegalCaseReports Summ (https://archive.ics.uci.edu/ml/machine-learning-databases/00239, https://aclanthology.org/W12-0515.pdf) ==> no re-destribution allowed
# Indian/Australian Summarization (https://github.com/manavkapadnis/LegalEvaluation_LREC2022) ==> too long and for australian data, annotation done automatically
# BVACItationPrediction (https://github.com/TUMLegalTech/bva-citation-prediction) ==> no dataset downloadable directly
# BSARD (https://github.com/maastrichtlawtech/bsard) ==> legal articles are not available directly
# EurLexSum (https://huggingface.co/datasets/dennlinger/eur-lex-sum) ==> very long texts and summaries

# TODO: Tasks still to add
"""
Contract extraction dataset (http://nlp.cs.aueb.gr/software_and_datasets/CONTRACTS_ICAIL2017/index.html, http://nlp.cs.aueb.gr/pubs/icail2017.pdf)
Cornell eRulemaking Corpus (https://facultystaff.richmond.edu/~jpark/data/jpark_lrec18.zip, https://facultystaff.richmond.edu/~jpark/papers/jpark_lrec18.pdf)
German Rental Agreements (https://github.com/sebischair/Legal-Sentence-Classification-Datasets-and-Models, https://www.researchgate.net/publication/332171940_Classifying_Semantic_Types_of_Legal_Sentences_Portability_of_Machine_Learning_Models)
US Caselaw Segmentation (https://github.com/jsavelka/us-dec-func-iss-sgm/blob/master/trade_secret_cases.json, http://ebooks.iospress.nl/volumearticle/50840)
Cookie Policy Summarization (https://github.com/senjed/Summarization-of-Privacy-Policies, http://ceur-ws.org/Vol-2645/paper3.pdf)
BVA Summarization (https://github.com/luimagroup/bva-summarization, https://dl.acm.org/doi/10.1145/3322640.3326728)
Australian Case Citation Summarization (https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports)

Arya: 
LegalLinking (https://github.com/mayhewsw/legal-linking)
Privacy Policies Summarization (https://github.com/senjed/Summarization-of-Privacy-Policies
E-NER (https://github.com/terenceau2/E-NER-Dataset)
GerDALIR (https://github.com/lavis-nlp/GerDaLIR)
Dutch Legal Summarization (https://github.com/prijsdf/dutch-legal-summarization)
Covid Law Matching (https://github.com/DFKI-NLP/covid19-law-matching)
BUILD (https://legal-nlp-ekstep.github.io/Competitions/Rhetorical-Role/)
CASS (https://github.com/euranova/CASS-dataset)
ECHR Argument Mining (http://www.di.uevora.pt/~pq/echr/)
Greek NER (https://github.com/nmpartzio/elNER)
Indian NER (https://arxiv.org/pdf/2211.03442.pdf, https://github.com/Legal-NLP-EkStep/legal_NER/tree/main/representative_judgments_sample)
LawngNLI (https://arxiv.org/pdf/2212.03222.pdf)
Privacy Policies (https://usableprivacy.org/data) (excluding OPP-115 Corpus: already present in natural instructions)
MakeThisYourLastTime (https://www.makethisyourlasttime.com/essay-bank/)
"""

NER_DELIMITER = "|"


def get_ner_instruction(ner_tags):
    return f"Predict the named entity types for each token (delimited by '{NER_DELIMITER}'). " \
           f"The named entities are: {' '.join(ner_tags)}."


def build_ner_answer(tokens, tags):
    f"Sentence: {NER_DELIMITER.join(tokens)}\n\n" \
    f"Named Entity Types: {NER_DELIMITER.join(tags)}\n\n"


def build_summarization_answer(input, summary):
    return f"Passage: {input}. Summary: {summary}"


def get_multiple_choice_instruction_bank():
    return ["Please answer these multiple choice questions. Denote the correct answer as \"Answer\".",
            "Pick the most likely correct answer."]


output_file_idx = 0
category = "law_instruct"
train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")

# TODO maybe do not use xP3 and natural instructions but only code and legal instructions becuase of figure 4: https://arxiv.org/pdf/2210.11416v5.pdf

# TODO always check if current file is too large and then save to next one

# TODO create file for each task type (summarization, qa, etc.) and add it as a column in the jsonl file

# TODO save instructions and answers into different columns for MT

# TODO do not use MT for basic training but only for fine-tuning

# swiss_judgment_prediction is handled separately
print("############################")
print("########## joelito/lextreme ###########")
print("############################")
source = "https://huggingface.co/datasets/joelito/lextreme"

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
    "mapa_course": [
        'O', 'B-ORGANISATION', 'I-ORGANISATION', 'B-ADDRESS', 'I-ADDRESS', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON',
        'B-AMOUNT', 'I-AMOUNT', 'B-TIME', 'I-TIME'
    ],
    "mapa_fine": [
        'O', 'B-BUILDING', 'I-BUILDING', 'B-CITY', 'I-CITY', 'B-COUNTRY', 'I-COUNTRY', 'B-PLACE', 'I-PLACE',
        'B-TERRITORY', 'I-TERRITORY', 'I-UNIT', 'B-UNIT', 'B-VALUE', 'I-VALUE', 'B-YEAR', 'I-YEAR',
        'B-STANDARD ABBREVIATION', 'I-STANDARD ABBREVIATION', 'B-MONTH', 'I-MONTH', 'B-DAY', 'I-DAY', 'B-AGE', 'I-AGE',
        'B-ETHNIC CATEGORY', 'I-ETHNIC CATEGORY', 'B-FAMILY NAME', 'I-FAMILY NAME', 'B-INITIAL NAME', 'I-INITIAL NAME',
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
    "greek_legal_code_chapter_level": "In this task, you are given a Greek legislative document. "
                                      "Predict the chapter level category of the 'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    "greek_legal_code_subject_level": "In this task, you are given a Greek legislative document. "
                                      "Predict the subject level category of the 'Permanent Greek Legislation Code - Raptarchis (Ραπτάρχης)' the document belongs to.",
    "greek_legal_code_volume_level": "In this task, you are given a Greek legislative document. "
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
    "mapa_ner_coarse_grained": "In this task, you are given a sentence from the EUR-Lex database. "
                               "Predict the coarse grained named entity type for each token.",
    "mapa_ner_fine_grained": "In this task, you are given a sentence from the EUR-Lex database. "
                             "Predict the fine grained named entity type for each token.",
}

TASK_CODE_MAPPING = {
    'brazilian_court_decisions_judgment': 'SLTC',
    'brazilian_court_decisions_unanimity': 'SLTC',
    'german_argument_mining': 'SLTC',
    'greek_legal_code_chapter_level': 'SLTC',
    'greek_legal_code_subject_level': 'SLTC',
    'greek_legal_code_volume_level': 'SLTC',
    'online_terms_of_service_unfairness_levels': 'SLTC',
    'online_terms_of_service_clause_topics': 'MLTC',
    'covid19_emergency_event': 'MLTC',
    'multi_eurlex_level_1': 'MLTC',
    'multi_eurlex_level_2': 'MLTC',
    'multi_eurlex_level_3': 'MLTC',
    'greek_legal_ner': 'NER',
    'legalnero': 'NER',
    'lener_br': 'NER',
    'mapa_ner_coarse_grained': 'NER',
    'mapa_ner_fine_grained': 'NER',
}

for subset, instructions in instructions_for_subsets.items():
    dataset = load_dataset("joelito/lextreme", subset)["train"]
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
                answers = [(f"Passage {input_text} Labels: {','.join(correct_labels)}", lang) for lang, text in
                           input_text.items()]
            else:
                answers = [(f"Passage {input_text} Labels: {','.join(correct_labels)}", example['language'])]

        elif task_code == 'NER':
            answers = [(build_ner_answer(example["input"], correct_labels), example['language'])]

        for answer, lang in answers:
            datapoint = f"{instructions}\n\n{answer}"
            if os.path.getsize(get_output_file_name(category, output_file_idx)) > MAX_FILE_SIZE:
                train_f.close()
                output_file_idx += 1
                train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")
            write_json_line(train_f, datapoint, lang, source)

# case_hold is already in natural instructions
print("############################")
print("########## lex_glue ###########")
print("############################")
source = "https://huggingface.co/datasets/lex_glue"

instructions_for_subsets = {
    "ecthr_a": "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). "
               "Predict the articles of the ECtHR that were violated (if any).",
    "ecthr_b": "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). "
               "Predict the articles of ECtHR that were allegedly violated (considered by the court).",
    "scotus": "In this task, you are given a case heard at the Supreme Court of the United States (SCOTUS). "
              "Predict the relevant issue area.",
    "eurlex": "In this task, you are given an EU law document published in the EUR-Lex portal. "
              "Predict the relevant EuroVoc concepts.",
    "ledgar": "In this task, you are given a contract provision from contracts obtained from US Securities and Exchange Commission (SEC) filings."
              "Predict the main topic.",
    "unfair_tos": "In this task, you are given a sentence from a Terms of Service (ToS) document from on-line platforms. "
                  "Predict the types of unfair contractual terms",
}

TASK_CODE_MAPPING = {
    'ecthr_a': 'MLTC',
    'ecthr_b': 'MLTC',
    'scotus': 'SLTC',
    'eurlex': 'MLTC',
    'ledgar': 'SLTC',
    'unfair_tos': 'MLTC',
}

for subset, instructions in instructions_for_subsets.items():
    dataset = load_dataset("lex_glue", subset)["train"]
    task_code = TASK_CODE_MAPPING[subset]

    if task_code == 'SLTC':
        class_label = dataset.features["label"]

    for example in dataset:
        # get correct labels
        if task_code == 'SLTC':
            correct_label = class_label.int2str(example['label'])  # get label name for correct label
            correct_labels = correct_label if isinstance(correct_label, list) else [correct_label]
        elif task_code == 'MLTC':
            correct_labels = list(map(str, example['labels']))  # here we don't have any mapping to label names

        input_text = example['input']
        if 'ecthr' in subset:
            input_text = " ".join(input_text)
        answer = f"Passage {input_text} Labels: {','.join(correct_labels)}"

        datapoint = f"{instructions}\n\n{answer}"
        if os.path.getsize(get_output_file_name(category, output_file_idx)) > MAX_FILE_SIZE:
            train_f.close()
            output_file_idx += 1
            train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")
        write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## Swiss Judgement Prediction ###########")
print("############################")
x = load_dataset('swiss_judgment_prediction', 'all+mt')['train']
source = "https://huggingface.co/datasets/swiss_judgment_prediction"
for example in x:
    court_location = "" if example['region'] == "n/a" else f"The court is located in {example['region']}."
    judgement = ["dismiss", "approve"][example['label']]
    datapoint = f"Determine if you think the Swiss court will dismiss or approve the case. {court_location}\n\nFacts:{example['text']}\nJudgement: {judgement}"
    write_json_line(train_f, datapoint, example["language"], source)

    datapoint = f"What area of law is this case related to?\n\nCase:{example['text']}\nArea of Law: {example['legal area']}"
    write_json_line(train_f, datapoint, example["language"], source)

    if court_location != "":
        datapoint = f"Where do you think this case was adjudicated?\n\nCase:{example['text']}\nRegion: {example['region']}"
        write_json_line(train_f, datapoint, example["language"], source)

    outcome_mc1 = ["(a)", "(b)"][example["label"]]
    text = example['text']
    datapoint = f"{random.choice(get_multiple_choice_instruction_bank())}\n\n" \
                f"Question: {text} How would the court find?\n(a) The court should dismiss the case.\n(b) The court should affirm the case.\n" \
                f"Answer: {outcome_mc1}."
    write_json_line(train_f, datapoint, example["language"], source)

    outcome_mc1 = ["(b)", "(a)"][example["label"]]
    datapoint = f"{random.choice(get_multiple_choice_instruction_bank())}\n\n" \
                f"Question: {text} How would the court find?\n(a) The court should approve the case.\n(b) The court should dismiss the case.\n" \
                f"Answer: {outcome_mc1}."
    write_json_line(train_f, datapoint, example["language"], source)

print("############################")
print("########## BrCAD-5 ###########")
print("############################")
# load locally because when loading from the hub I get the following weird error: TypeError: Couldn't cast array of type double to null
x = load_dataset('json', 'raw_data/brcad_5/train.jsonl.xz')['train']
source = "https://huggingface.co/datasets/joelito/BrCAD-5"

for example in x:
    text = example['preprocessed_full_text_first_instance_court_ruling']
    datapoint = f"Determine what you think the Brazilian appeals court will rule for the case.\n\nCase:{text}\nJudgement: {example['label']}"
    write_json_line(train_f, datapoint, "pt", source)

    datapoint = f"What area of law is this case related to?\n\nCase:{text}\nArea of Law: {example['current_case_class']}"
    write_json_line(train_f, datapoint, "pt", source)

    for level in ["1st", "2nd", "3rd"]:
        datapoint = f"What {level}-level topic is this case related to?\n\nCase:{text}\nTopic: {example[f'case_topic_{level}_level']}"
        write_json_line(train_f, datapoint, "pt", source)

    outcome_mc1 = ["(a)", "(b)"][['NÃO PROVIMENTO', 'PROVIMENTO'].index(example["label"])]
    text = example['text']
    datapoint = f"{random.choice(get_multiple_choice_instruction_bank())}\n\n" \
                f"Question: {text} How would the court find?\n(a) The court should dismiss the case.\n(b) The court should affirm the case.\n" \
                f"Answer: {outcome_mc1}."
    write_json_line(train_f, datapoint, "pt", source)

    outcome_mc1 = ["(b)", "(a)"][['NÃO PROVIMENTO', 'PROVIMENTO'].index(example["label"])]
    datapoint = f"{random.choice(get_multiple_choice_instruction_bank())}\n\n" \
                f"Question: {text} How would the court find?\n(a) The court should approve the case.\n(b) The court should dismiss the case.\n" \
                f"Answer: {outcome_mc1}."
    write_json_line(train_f, datapoint, "pt", source)

print("############################")
print("########## MultiLexSum ###########")
print("############################")
source = "https://huggingface.co/datasets/allenai/multi_lexsum"
df = load_dataset("allenai/multi_lexsum")["train"]

instruction_bank = [
    "Summarize the following summary of a US legal document further. ",
    "Consider the summary of a US legal document and summarize it further. "]
for example in df:
    input = example["summary/long"]
    if example["summary/short"]:
        summary = example["summary/short"]
        datapoint = f"{random.choice(instruction_bank)}\n\n{build_summarization_answer(input, summary)}"
        write_json_line(train_f, datapoint, "en", source)
    if example["summary/tiny"]:
        summary = example["summary/tiny"]
        datapoint = f"{random.choice(instruction_bank)}\n\n{build_summarization_answer(input, summary)}"
        write_json_line(train_f, datapoint, "en", source)
    if example["summary/short"] and example["summary/tiny"]:
        input = example["summary/short"]
        summary = example["summary/tiny"]
        datapoint = f"{random.choice(instruction_bank)}\n\n{build_summarization_answer(input, summary)}"
        write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## LegalCaseDocumentSummarization ###########")
print("############################")
source = "https://huggingface.co/datasets/joelito/legal_case_document_summarization"
df = load_dataset("joelito/legal_case_document_summarization")["train"]


def get_instruction_bank(court):
    return [
        f"Summarize the document of the {court}. ",
        f"Consider the document of the {court} and summarize it. "
    ]


for example in df:
    if "IN" in example["dataset_name"]:
        instruction_bank = get_instruction_bank("Indian Supreme Court case")
    elif "UK" in example["dataset_name"]:
        instruction_bank = get_instruction_bank("U.K. Supreme Court case")
    else:
        continue
    input = example["judgement"]
    summary = example["summary/full"]
    datapoint = f"{random.choice(instruction_bank)}\n\n{build_summarization_answer(input, summary)}"
    write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## PlainEnglishContractsSummarization ###########")
print("############################")
source = "https://huggingface.co/datasets/joelito/plain_english_contracts_summarization"
df = load_dataset("joelito/plain_english_contracts_summarization")["train"]


def get_instruction_bank(document):
    return [
        f"Summarize the following excerpt of a {document} document. ",
        f"Consider the excerpt of a {document} document and summarize it. "
    ]


for example in df:
    instruction_bank = get_instruction_bank(example["doc"])
    input = example["original_text"]
    summary = example["reference_summary"]
    datapoint = f"{random.choice(instruction_bank)}\n\n{build_summarization_answer(input, summary)}"
    write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## German-LER ###########")
print("############################")
source = "https://huggingface.co/datasets/elenanereiss/german-ler"
df = load_dataset("elenanereiss/german-ler")["train"]

ner_fine_tags = ['B-AN', 'B-EUN', 'B-GRT', 'B-GS', 'B-INN', 'B-LD', 'B-LDS', 'B-LIT', 'B-MRK', 'B-ORG', 'B-PER', 'B-RR',
                 'B-RS', 'B-ST', 'B-STR', 'B-UN', 'B-VO', 'B-VS', 'B-VT', 'I-AN', 'I-EUN', 'I-GRT', 'I-GS', 'I-INN',
                 'I-LD', 'I-LDS', 'I-LIT', 'I-MRK', 'I-ORG', 'I-PER', 'I-RR', 'I-RS', 'I-ST', 'I-STR', 'I-UN', 'I-VO',
                 'I-VS', 'I-VT', 'O']
ner_coarse_tags = ['B-LIT', 'B-LOC', 'B-NRM', 'B-ORG', 'B-PER', 'B-REG', 'B-RS', 'I-LIT', 'I-LOC', 'I-NRM', 'I-ORG',
                   'I-PER', 'I-REG', 'I-RS', 'O'],

introduction_sentence = "Consider the following sentence from a German federal court decision."
instruction_bank_fine = [f"{introduction_sentence} {get_ner_instruction(ner_fine_tags)}", ]
instruction_bank_coarse = [f"{introduction_sentence} {get_ner_instruction(ner_coarse_tags)}"]
for example in df:
    datapoint = f"{random.choice(instruction_bank_fine)}\n\n{build_ner_answer(example['tokens'], example['ner_tags'])}"
    write_json_line(train_f, datapoint, "de", source)

    datapoint = f"{random.choice(instruction_bank_coarse)}\n\n{build_ner_answer(example['tokens'], example['ner_coarse_tags'])}"
    write_json_line(train_f, datapoint, "de", source)

print("############################")
print("########## Mining Legal Arguments ###########")
print("############################")


def get_all_ner_labels(df, labels_column_name="labels"):
    all_labels = set()
    for example in df:
        all_labels.update(example[labels_column_name])
    return all_labels


for type in ["agent", "argType"]:
    source = f"https://huggingface.co/datasets/joelito/mining_legal_arguments_{type}"
    df = load_dataset(f"joelito/mining_legal_arguments_{type}")["train"]
    all_labels = get_all_ner_labels(df)
    introduction_sentence = "Consider the following sentence from an ECtHR decision. "
    instruction_bank = [f"{introduction_sentence} {get_ner_instruction(all_labels)}", ]
    for example in df:
        datapoint = f"{random.choice(instruction_bank)}\n\n{build_ner_answer(example['tokens'], example['labels'])}"
        write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## Contract-NLI ###########")
print("############################")
source = "https://huggingface.co/datasets/kiddothe2b/contract-nli"
df = load_dataset("kiddothe2b/contract-nli")["train"]

class_label = df.features["label"]
instruction_bank = [
    "Consider the following Contract Passage and Hypothesis. Predict whether the Contract Passage entails/contradicts/is neutral to the Hypothesis (entailment, contradiction or neutral).",
    "Does the following Contract Passage entail/contradict/stand neutral to the Hypothesis?"]
for example in df:
    datapoint = f"{random.choice(instruction_bank)}\n\n" \
                f"Contract Passage: {example['premise']}\n\n" \
                f"Hypothesis: {example['hypothesis']}\n\n" \
                f"Entailment: {class_label.int2str(example['label'])}"
    write_json_line(train_f, datapoint, "en", source)

# Add math-type reasoning b/c tax has that flavor
print("############################")
print("########## gsm8k ###########")
print("############################")
source = "https://huggingface.co/datasets/gsm8k"
x = load_dataset("gsm8k", "main", split="train")

instruction_bank = ["Answer the question, make sure to show your work.",
                    "Answer the math question step by step. Show your work.",
                    "Answer the following question in logical steps.",
                    "Answer the following questions."]
for example in x:
    datapoint = f"{random.choice(instruction_bank)}\n\nQ: {example['question']}\nA: {example['answer']}"
    if os.path.getsize(get_output_file_name(category, output_file_idx)) > MAX_FILE_SIZE:
        train_f.close()
        output_file_idx += 1
        train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")
    write_json_line(train_f, datapoint, "en", source)

x = load_dataset("gsm8k", "socratic", split="train")

instruction_bank = ["Answer the question, make sure to ask yourself follow up questions.",
                    "Answer the math question using the socratic method. Show your work.",
                    "Answer the following question in logical steps.",
                    "Answer the following questions. Make sure to ask any follow up questions as needed."]
for example in x:
    datapoint = f"{random.choice(instruction_bank)}\n\nQ: {example['question']}\nA: {example['answer']}"
    if os.path.getsize(get_output_file_name(category, output_file_idx)) > MAX_FILE_SIZE:
        train_f.close()
        output_file_idx += 1
        train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")
    write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## Lila ###########")
print("############################")
json_files = [pos_json for pos_json in os.listdir("raw_data/all_lila/") if pos_json.endswith('.json')]
instruction_bank = ["Consider the following question. Write a Python program to solve it.",
                    "Write a Python program to solve the following question, denote it as \"Program:\". Provide the output as \"Answer:\"."]
for json_file in json_files:
    with open(os.path.join("raw_data/all_lila/", json_file), "r") as f:
        loaded_file = json.loads(f.read())
        for example in loaded_file["Instances"]:
            if example["split"] != "train":
                continue
            for program, answer in zip(example['Output Program'], example['Output Answer']):
                datapoint = f"{random.choice(instruction_bank)}\n\nQuestion: {example['Input']}\nProgram:\n```python\n{program}\n```\nAnswer: {answer}"
                write_json_line(train_f, datapoint, "en", "https://github.com/allenai/Lila")

print("############################")
print("########## Sara Prolog ###########")
print("############################")
# TODO do we have an url here?
source = "sara"
instruction_bank = ["Convert the following statute into prolog code.",
                    "Write a prolog program to convert the statute into code, denote it as \"Prolog Program:\"."]
json_files = [pos_json for pos_json in os.listdir("raw_data/sara_statutes/source")]
for json_file in json_files:
    with open(os.path.join("raw_data/sara_statutes/source/", json_file), "r") as f_normal:
        with open(os.path.join("raw_data/sara_statutes/prolog/", json_file) + ".pl", "r") as f_prolog:
            datapoint = f"{random.choice(instruction_bank)}\n\nStatute:\n{f_normal.read()}\n\nProlog Program:\n\n{f_prolog.read()}"
            write_json_line(train_f, datapoint, "en", source)

instruction_bank = ["Convert the following fact pattern into prolog code. Then answer the question.",
                    "Write a prolog program to mark all the facts, denote it as \"Prolog Program:\". Then answer the question, denote your answer as \"Answer\"."]
json_files = [pos_json for pos_json in os.listdir("raw_data/sara_cases/") if pos_json != "train"]
with open("raw_data/sara_cases/train", "r") as train_list_f:
    train_list = [x.strip() for x in train_list_f.readlines()]
for json_file in json_files:
    with open(os.path.join("raw_data/sara_cases/", json_file), "r") as f_normal:
        if json_file.split(".pl")[0] not in train_list:
            print(f"Skipping {json_file}")
        text = f_normal.read()

        facts_and_question = text.split("% Facts")[0]
        program = text.split("% Facts")[1]

        if "Entailment" in facts_and_question:
            answer = "True"
        elif "Contradiction" in facts_and_question:
            answer = "False"
        else:
            answer = facts_and_question.split("% Question")[1].split("?")[-1].strip()

        facts_and_question = facts_and_question.replace("Entailment", "Is this True or False?")
        facts_and_question = facts_and_question.replace("Contradiction", "Is this True or False?")
        facts_and_question = facts_and_question.replace("\n", " ")
        facts_and_question = facts_and_question.replace("% Text", "Facts:")
        facts_and_question = facts_and_question.replace("% Question", "\nQuestion:")
        facts_and_question = facts_and_question.replace("%", "").strip()

        datapoint = f"{random.choice(instruction_bank)}\n\n{facts_and_question}\n\nProlog Program:\n{program.strip()}\nAnswer: {answer}"
        write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## Civipro Questions ###########")
print("############################")
instruction_bank_generate_questions_from_passage = [
    "Consider these questions about American civil procedure. Given the provided information answer them to the best of your ability.",
    "Here is some information that can help you answer the question. Provide an analysis of the options and then pick the correct answer.",
    "Given a passage of text about Civil Procedure in the United States, generate a question that can be answered by reading the passage.",
    "Generate a CivPro question that can be answered by reading the passage, denote it as \"Question:\", provide an answer as \"Answer\"."]
instruction_bank_generate_questions_no_passage = [
    "Consider these questions about American civil procedure. Answer them to the best of your ability, first provide an explanation then the answer.",
    "Consider these civil procedure questions. Provide an analysis of the options and then pick the correct answer.",
    "Answer this Civil Procedure question based on law in the United States. Provide an explanation first.",
    "Answer this CivPro question provide an answer as \"Answer\", but first provide an explanation as \"Explanation\"."]
instruction_bank_generate_questions_no_explanation = [
    "Consider these questions about American civil procedure. Answer them to the best of your ability, DO NOT provide an explanation before giving the answer.",
    "Consider these civil procedure questions. Pick the correct answer.",
    "Given a passage of text about Civil Procedure in the United States, answer the question.",
    "Answer this CivPro question provide an answer as \"Answer\"."]

df = pd.read_csv("./raw_data/civpro_questions_train.csv")

questions_dict = defaultdict(dict)

for idx, row in tqdm(df.iterrows()):
    if row["question"] not in questions_dict:
        questions_dict[row["question"]] = {"choices": []}
    questions_dict[row["question"]]["choices"].append((row["answer"], row["label"], row["analysis"]))
    questions_dict[row["question"]]["explanation_passage"] = row["explanation"]
    questions_dict[row["question"]]["chain_of_thought"] = row["complete analysis"]

for question, values in questions_dict.items():
    choices = values["choices"]
    question = ".".join(question.split(".")[1:])
    if len(choices) < 4:
        print(f"Skipping {question} because it has less than 2 choices")
        continue
    # random.shuffle(choices)
    lookup = ["A", "B", "C", "D", "E", "F", "G"]
    analysis_string = values['chain_of_thought']  # "\n".join([f"{choice[2]}" for i, choice in enumerate(choices)])
    try:
        choice_string = "\n".join([f"{lookup[i]}. {choice[0]}" for i, choice in enumerate(choices)])
        correct_answer = lookup[[idx for idx, choice in enumerate(choices) if choice[1] == 1][0]]
    except:
        print(f"Skipping {question} because of some problem.")
        continue
    datapoint_with_passage = f"{random.choice(instruction_bank_generate_questions_from_passage)}\n\n{values['explanation_passage']}\n\nQuestion: {question}\n{choice_string}\nAnswer: {correct_answer}"
    datapoint_no_passage = f"{random.choice(instruction_bank_generate_questions_no_passage)}\n\nQuestion: {question}\n{choice_string}\nExplanation: {analysis_string}\nAnswer: {correct_answer}"
    datapoint_no_explanation = f"{random.choice(instruction_bank_generate_questions_no_explanation)}\n\nQuestion: {question}\n{choice_string}\nAnswer: {correct_answer}"

    for datapoint in [datapoint_no_passage, datapoint_no_explanation, datapoint_with_passage]:
        write_json_line(train_f, datapoint, "en", "civpro_questions")  # TODO do we have an url here?

# The first 1200 are extra bar exam questions, not sure if we want to keep these in
print("############################")
print("########## Professional Law ###########")
print("############################")
instructions_examples = ["Generate some Multistate Bar Exam questions according to U.S. law.",
                         "Answer these legal questions. Use American Law. A few examples are provided first to give the answer format.",
                         "Answer these U.S. Multistate Bar Exam questions. A few examples are provided first to give the answer format.",
                         "Pick the most correct option considering U.S. Law."]
instructions_zero_shot = ["Answer these legal questions. Use American Law. Provide the choice as \"Answer:\"",
                          "Answer these U.S. Multistate Bar Exam questions. Provide the choice as \"Answer:\"",
                          "Pick the most correct option considering U.S. Law. Output the choice as \"Answer:\""]
# TODO do we really want this now?
# TODO is this the same as MMMLU?
df = load_dataset("hendrycks_test", "professional_law", split="auxiliary_train").select(range(1200))


def shuffle_choices(choices: List[str], answer: int):
    x = list(enumerate(choices))
    random.shuffle(x)
    indices, choices = zip(*x)
    answer = indices.index(answer)
    return choices, answer


source = "auxiliary_train_hendrycks_test"
for i, (this_question, this_choices, this_answer) in tqdm(enumerate(zip(
        df["question"], df["choices"], df["answer"]
)), total=len(df)):
    prompt_samples = df.select(random.sample(list(range(0, i)) + list(range(i + 1, len(df))), 3))
    prompt = ""
    for j, (prompt_question, prompt_choices, prompt_answer) in enumerate(zip(
            prompt_samples["question"], prompt_samples["choices"], prompt_samples["answer"]
    )):
        prompt += f"Question: {prompt_question}\n"
        lookup = ["(a)", "(b)", "(c)", "(d)"]
        prompt_choices, prompt_answer = shuffle_choices(prompt_choices, prompt_answer)
        for i, choice in enumerate(prompt_choices):
            prompt += f"{lookup[i]} {choice}\n"
        prompt += (
            f"The Final Answer: {lookup[prompt_answer]}\n\n"
        )
        prompt += "###\n\n"

    cur_question = prompt
    cur_question += f"Question: {this_question}\n"
    for i, choice in enumerate(this_choices):
        lookup = ["(a)", "(b)", "(c)", "(d)"]
        cur_question += f"{lookup[i]} {choice}\n"

    cur_question += (
        f"The Final Answer: {lookup[this_answer]}"
    )
    datapoint = cur_question

    final_datapoint = random.choice(instructions_examples) + "\n\n" + datapoint
    write_json_line(train_f, final_datapoint, "en", source)

    datapoint_zero_shot = datapoint.replace("The Final Answer: ", "Answer: ").split("###")[-1].strip()
    final_datapoint_zero_shot = random.choice(instructions_zero_shot) + "\n\n" + datapoint_zero_shot
    write_json_line(train_f, final_datapoint_zero_shot, "en", source)

print("############################")
print("########## MBE ###########")
print("############################")
# TODO do we have an url for the source here?
source = "MBE"
df = pd.read_csv("raw_data/mbe_train.csv")
instructions_examples = [
    "Answer these legal questions. Use American Law. Please explain your thought process and then answer the question.",
    "Answer these U.S. Multistate Bar Exam questions. Please provide an explanation first.",
    "Pick the most correct option considering U.S. Law. Explain your answer first."]
instruction_bank_subject = [
    "What subject of U.S. law is this question about? Pick one from: TORTS, CONTRACTS, CRIM. LAW, EVIDENCE, CONST. LAW, REAL PROP.",
    "What area of American law is this question about? Pick one from: TORTS, CONTRACTS, CRIM. LAW, EVIDENCE, CONST. LAW, REAL PROP."]
instruction_bank_subject_generation = [
    "Generate a bar exam multiple choice question, along with an explanation and answer, for the following subject: ",
    "Generate an MBE MC question for "]
for idx, row in df.iterrows():
    # source_year = row["Source"].split("MBE-")[1].split("-")[0]
    if isinstance(row['Prompt'], str) and row['Prompt'].strip() != "" and row['Prompt'].strip() != "nan":
        question = row['Prompt']
    else:
        question = ""
    question += f" {row['Question']}"
    question = question.strip()
    choices = [row["Choice A"], row["Choice B"], row["Choice C"], row["Choice D"]]
    answer = row["Answer"]
    subject = row["Subject"]
    positive_passage = row["Positive Passage"]
    datapoint = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        lookup = ["A", "B", "C", "D"]
        datapoint += f"{lookup[i]}. {choice}\n"
    data_no_answer = datapoint
    datapoint += f"Explanation: {positive_passage}\nAnswer: {answer}"
    datapoint_with_answer = datapoint
    # if source_year.strip() != "" and int(source_year) > 1950 and int(source_year) < 2023:
    #     source_year_string = f" Consider only the law and relevant cases before {source_year}."
    # else:
    #     source_year_string = ""
    final_datapoint = random.choice(instructions_examples) + "\n\n" + datapoint
    write_json_line(train_f, final_datapoint, "en", source)

    if isinstance(subject, str) and subject.strip() != "":
        datapoint = f"{random.choice(instruction_bank_subject)}\n\n{data_no_answer}\nSubject: {subject}"

        datapoint = random.choice(instructions_examples) + "\n\n" + datapoint
        write_json_line(train_f, datapoint, "en", source)

        datapoint = random.choice(instruction_bank_subject_generation) + subject + ".\n\n" + datapoint_with_answer
        write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## Littleton ###########")
print("############################")
json_files = [pos_json for pos_json in os.listdir("raw_data/littleton/examples/") if pos_json.endswith('.json')]
instruction_bank = [
    "Consider the law of future interests and conveyances in American property law. Consider the chain of events and then state the interests.",
    "According to American law, consider the chain of events and future interests."]
source = "https://github.com/grimmelm/littleton"
for json_file in json_files:
    with open(os.path.join("raw_data/littleton/examples/", json_file), "r") as f:
        loaded_file = json.loads(f.read())[1]
        if isinstance(loaded_file, str):
            continue
        for example in loaded_file["examples"]:
            datapoint = f"{random.choice(instruction_bank)}\n\nEvents: {example['program']}\nAnswer: {example['result']}"
            write_json_line(train_f, datapoint, "en", source)

json_files = [pos_json for pos_json in os.listdir("raw_data/littleton/tests/edwards") if pos_json.endswith('.toml')]
instruction_bank = [
    "Consider the law of future interests and conveyances in American property law. Consider the chain of events and then output a graph structure representing the events.",
    "According to American law, consider the chain of events and future interests. Output a graph structure representing the events and any interests."]
for json_file in json_files:
    with open(os.path.join("raw_data/littleton/tests/edwards/", json_file), "r") as f:
        loaded_file = toml.loads(f.read())
        for example in loaded_file["tests"]:
            if "expected" not in example:
                continue
            datapoint = f"{random.choice(instruction_bank)}\n\nEvents: {example['program']}\nAnswer: {example['expected']}"
            write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## JEC-QA ###########")
print("############################")
instruction_bank = [
    "Answer these multiple choice reasoning questions about Chinese Law. Select all answers that apply, you may have multiple correct answers.",
    "Answer these Chinese Law multiple choice questions, you might have multiple correct answers. Denote your answer(s) as \"Answer: [answer(s)].\""]

with open("./raw_data/jecqa_0_train.json") as f:
    questions = [json.loads(x) for x in f.readlines()]
    with open("./raw_data/jecqa_1_train.json") as f:
        questions.extend([json.loads(x) for x in f.readlines()])

for q in questions:
    prompt = f"random.choice(instruction_bank)\n\n{q['statement']}\n\n"
    for k, v in q["option_list"].items():
        prompt += f"{k}. {v}\n"
    prompt += "\n\nFinal Answer(s): {','.join(q['answer'])}"
    write_json_line(train_f, prompt, "zh", "https://jecqa.thunlp.org/")

# Legal Judgement Prediction: US Class Actions
print("############################")
print("########## darrow-ai/USClassActions ###########")
print("############################")
df = load_dataset("darrow-ai/USClassActions")["train"]
source = "https://huggingface.co/datasets/darrow-ai/USClassActions"
instruction_bank = [
    "Read the following United States class action complaint. Predict whether the complaint will be won or not. Output \"win\" or \"lose\".",
    "Will this class action complaint be successful in U.S. Court?"]
for example in df:
    datapoint = f"{random.choice(instruction_bank)}\n\n{example['target_text']}\n\nLikely Verdict: {example['verdict']}"
    if os.path.getsize(get_output_file_name(category, output_file_idx)) > MAX_FILE_SIZE:
        train_f.close()
        output_file_idx += 1
        train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")
    write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## Short Answer Feedback (SAF) ###########")
print("############################")

df = load_dataset("JohnnyBoy00/saf_legal_domain_german")

instruction_bank_openqa = ["Consider this question in the context of German law. Provide the correct reference answer.",
                           "Answer the question about German law. Make sure it is correct."]
instruction_bank_feedback = [
    "Here is a question and answer pair related to German Law. Considering the student provided answer, provide detailed feedback and then provide a score of 1 for correct, 0.5 for partially correct, and 0 for incorrect.",
    "Consider the answer to the question, is it correct? Provide feedback and then give a score from 0 to 1.",
    "Consider the student's answer to the question. Rate it and provide feedback."]
instruction_error_class = [
    "Here is a question and answer pair related to German Law. Considering the student provided answer, provide detailed feedback and then provide a score of 1 for correct, 0.5 for partially correct, and 0 for incorrect.",
    "Consider the answer to the question, is it correct? Provide feedback and then give a score from 0 to 1. Note the error class.",
    "Consider the student's answer to the question. Rate it and provide feedback. Note the type of error."]

source = "https://huggingface.co/datasets/JohnnyBoy00/saf_legal_domain_german"
for example in df["train"]:
    datapoint = f"{random.choice(instruction_bank_openqa)}\n\nQ: {example['question']}\nA: {example['reference_answer']}"
    write_json_line(train_f, datapoint, "de", source)

    datapoint = f"{random.choice(instruction_bank_feedback)}\n\nQ: {example['question']}\nA: {example['provided_answer']}\nFeedback: {example['verification_feedback']}\nScore: {example['score']}"
    write_json_line(train_f, datapoint, "de", source)

    datapoint = f"{random.choice(instruction_error_class)}\n\nQ: {example['question']}\nA: {example['provided_answer']}\nFeedback: {example['verification_feedback']}\nScore: {example['score']}\nError Type: {example['error_class']}"
    write_json_line(train_f, datapoint, "de", source)

print("############################")
print("########## ILDC Dataset ###########")
print("############################")

df1 = pd.read_csv("raw_data/ILDC_multi.csv")
df1 = df1[df1["split"] == "train"]
df2 = pd.read_csv("raw_data/ILDC_single.csv")
df2 = df2[df2["split"] == "train"]

source = "https://github.com/Exploration-Lab/CJPE"
instruction_bank = [
    "According to Indian law, will this petition be accepted? If there is more than one petition consider whether the court will accept at least one.",
    "Will the court accept or reject this petition? Use Indian law. If there is more than one petition consider whether the court will accept at least one."]

for idx, row in df1.iterrows():
    decision = "Court Decision: Reject" if row["label"] == 0 else "Court Decision: Accept"
    datapoint = f"{random.choice(instruction_bank)}\n\n{row['text']}\n\n{decision}"
    write_json_line(train_f, datapoint, "en", source)

for idx, row in df2.iterrows():
    decision = "Court Decision: Reject" if row["label"] == 0 else "Court Decision: Accept"
    datapoint = f"{random.choice(instruction_bank)}\n\n{row['text']}\n\n{decision}"
    write_json_line(train_f, datapoint, "en", source)

# Scraped bar exam essays
print("############################")
print("########## CA Bar Exam Essays ###########")
print("############################")

source = "https://www.calbar.ca.gov/Admissions/Examinations/California-Bar-Examination/Past-Exams"
with open("raw_data/bar_exam_essays_ca.jsonl") as f:
    exams = [json.loads(x) for x in f.readlines()]
    for exam in exams:
        write_json_line(train_f, exam['text'], "en", source)

print("############################")
print("########## MC Exams Law ###########")
print("############################")

df = pd.read_csv("raw_data/raw_legal_mc_with_explanations.csv")

instruction_bank = ["Answer these questions according to the laws of the United States.",
                    "Pick the best answer according to U.S. law.",
                    "Pick the correct multiple choice answer according to American law."]
instruction_bank_expl = [
    "Answer these questions according to the laws of the United States. First explain your answer.",
    "Pick the best answer according to U.S. law. First explain your answer.",
    "Pick the correct multiple choice answer according to American law. Explain your answer then give the correct choice."]
for idx, row in df.iterrows():
    q, a, explanation, source = row["Question"], row["Answer"], row["Explanation"], row["Source"]

    # No chain of thought
    datapoint = f"{random.choice(instruction_bank)}\n\nQ:{q}\nA:{a}"
    write_json_line(train_f, datapoint, "en", source)

    # Chain of thought
    datapoint = f"{random.choice(instruction_bank_expl)}\n\nQ:{q}\nExplanation: {explanation}\nA:{a}"
    write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## Korean LegalQA ###########")
print("############################")

source = "https://raw.githubusercontent.com/haven-jeon/LegalQA/main/data/legalqa.jsonlines"
instruction_bank = ["Consider the following question. Retrieve the relevant Korean legal article.",
                    "What is the best South Korean law that can help answer this question.",
                    "What South Korean law best applies."]

with open("raw_data/legalqa.jsonlines", "r") as f:
    questions = [json.loads(x) for x in f.readlines()]

for question in questions:
    datapoint = f"{random.choice(instruction_bank)}\n\nQ: {question['question']}\nA: {question['answer']}"
    write_json_line(train_f, datapoint, "ko",
                    source)

print("############################")
print("########## Spanish Labor Law ###########")
print("############################")

df = pd.read_csv("raw_data/spanish_legal_qa.csv")
source = "https://zenodo.org/record/4256718#.Y5PoC7LMIlg"

instruction_bank = [
    "Consider this Spanish Labor Law translated passage. Answer the question using an extractive snippet of text.",
    "Consider this Spanish Labor Law translated passage. Answer the question from the context.",
    "Answer the following Spanish labor law question given the legal provision."]
for idx, row in df.iterrows():
    question, context, answer = row["Question"], row["context"], row["Answer text"]
    datapoint = f"{random.choice(instruction_bank)}\n\nContext: {context}\nQ: {question}\nA: {answer}"
    write_json_line(train_f, datapoint, "es", source)

# International citizenship law questions
print("############################")
print("########## International citizenship law questions ###########")
print("############################")

df1 = pd.read_csv("raw_data/data_v1.0_country-year-mode_acq.csv")
df2 = pd.read_csv("raw_data/data_v1.0_country-year-mode_loss.csv")
code_year = pd.read_csv("raw_data/data_v1.0_country-year.csv")
code_dictionary = pd.read_csv("raw_data/code_dictionary.csv")
source = "https://cadmus.eui.eu/handle/1814/73190"

for idx, row in df1.iterrows():
    mode_id = row["mode_id"]
    country = row["country"]
    law_article = row["article"]
    law_article = law_article.strip().replace('\n', ' ')
    specification = row["specification"]
    specification = specification.strip().replace('\n', ' ')
    if specification != "n.a.":
        specification = "The provision applies under the following conditions. " + specification
    else:
        specification = ""
    code_year_spec = code_year[code_year["country"] == country]
    code_year_spec = code_year_spec[f"{mode_id.strip()}_bin"].values[0]
    if code_year_spec == 99:
        code_year_spec = 0
    code_year_spec_answer = ["No.", "Yes."][code_year_spec]
    q = code_dictionary[code_dictionary["Mode ID"] == mode_id.strip()]["Focus"].values[0]
    if "No provision" in law_article:
        datapoint = f"Q: Consider the country of {country.strip()}. {q.strip()}\nA: {code_year_spec_answer} This is not covered in any provision."
    else:
        datapoint = f"Q: Consider the country of {country.strip()}. {q.strip()}\nA: {code_year_spec_answer} This is covered in: {law_article}. {specification}".strip()

    write_json_line(train_f, datapoint, "en", source)

for idx, row in df2.iterrows():
    mode_id = row["mode_id"]
    country = row["country"]
    law_article = row["article"]
    law_article = law_article.strip().replace('\n', ' ')

    specification = row["specification"]
    specification = specification.strip().replace('\n', ' ')
    if specification != "n.a.":
        specification = "The provision applies under the following conditions. " + specification
    else:
        specification = ""

    code_year_spec = code_year[code_year["country"] == country]
    code_year_spec = code_year_spec[f"{mode_id.strip()}_bin"].values[0]
    if code_year_spec == 99:
        code_year_spec = 0
    code_year_spec_answer = ["No.", "Yes."][code_year_spec]
    q = code_dictionary[code_dictionary["Mode ID"] == mode_id.strip()]["Focus"].values[0]
    if "No provision" in law_article:
        datapoint = f"Q: Consider the country of {country.strip()}. {q}\nA: {code_year_spec_answer} This is not covered in any provision."
    else:
        datapoint = f"Q: Consider the country of {country.strip()}. {q}\nA: {code_year_spec_answer} This is covered in: {law_article}. {specification}".strip()
    write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## EOIR PRIVACY ###########")
print("############################")

df = load_dataset("pile-of-law/eoir_privacy", "eoir_privacy", split="train")

instruction_bank = [
    "For each masked paragraph, determine if we should use a pseudonym for this case related to immigration law in the United States.",
    "Consider this paragraph from a precedential EOIR case. Should the IJ use a a pseudonym.",
    "Should the judge pseudonymize the person's name in this paragraph?"]

for example in df:
    lookup = ["Don't use pseudonym.", "Use pseudonym."]
    datapoint = f"{random.choice(instruction_bank)}\n\n{example['text']}\n{lookup[example['label']]}"
    write_json_line(train_f, datapoint, "en", "https://huggingface.co/datasets/pile-of-law/eoir_privacy")


# Will Validity
print("############################")
print("########## Valid Wills ###########")
print("############################")
instruction_bank = [
    "Given a statement in a will, the relevant U.S. law, is the condition supported, refuted, or unrelated.",
    "Is the statement in the will valid given the law and conditions? Answer with one of unrelated, supported, refuted."]
train = pd.read_csv('./raw_data/wills_train.csv', encoding='utf-8')  # replace with real path and dataset names
source = "https://arxiv.org/pdf/2210.16989.pdf"
for idx, row in train.iterrows():
    statement, conditions, law, classification = row["statement"], row["conditions"], row["law"], row["classification"]
    CLASSIFICATION_MAP = ['refuted', 'supported', 'unrelated']
    classification = CLASSIFICATION_MAP[classification]
    prompt = f"{random.choice(instruction_bank)}\n\nStatement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}\n\nAnswer: {classification}"
    prompt2 = f"Statement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}\n\nIs the statement supported by the law and condition?\n\nAnswer: {classification}"

    options_mc = ["supported", "refuted", "unrelated"]
    lookup = ["(a)", "(b)", "(c)"]
    random.shuffle(options_mc)
    option_mc_string = ""
    correct_option = None
    for choice_letter, option in zip(lookup, options_mc):
        if option == classification:
            correct_option = choice_letter
        option_mc_string += f"{choice_letter} {option}\n"
    prompt_mc = f"Statement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}\n\n{option_mc_string}\n\nAnswer: {correct_option}"
    write_json_line(train_f, prompt, "en", source)
    write_json_line(train_f, prompt2, "en", source)
    write_json_line(train_f, prompt_mc, "en", source)

# Chinese Bar Exam, no explanations.
print("############################")
print("########## LogiQA ###########")
print("############################")
instruction_bank = [
    "Answer these multiple choice reasoning questions about Chinese Law. There is only one right answer.",
    "Answer these Chinese Law multiple choice questions. There is only one correct answer. Denote your answer as \"Answer: [answer].\""]

with open("./raw_data/zh_train.txt", "r") as f:
    x = f.readlines()
    i = 0
    while True:
        blank = x[i]
        i += 1
        correct = x[i]
        i += 1
        context = x[i]
        i += 1
        question = x[i]
        i += 1
        choices = []
        for z in range(4):
            choices.append(x[i])
            i += 1
        datapoint = f"{random.choice(instruction_bank)}\n\nQuestion: {context.strip()} {question}{''.join(choices)}\n\nAnswer: ({correct.strip()})."
        write_json_line(train_f, datapoint, "zh", "https://github.com/lgw863/LogiQA-dataset")
        if i >= len(x): break

# ChangeMyView Argumentation
print("############################")
print("########## ChangeMyView ###########")
print("############################")
instruction_bank = ["You are given a position, create an argument that would change the original poster's mind.",
                    "Write a counter argument to the proposal.", "Write a counter argument to the r/changemyview post.",
                    "Write a counterargument to this reddit post."]
with open("./raw_data/train_pair_data.jsonlist") as f:
    x = [json.loads(s) for s in f.readlines()]
    for d in x:
        if isinstance(d['positive']['comments'][0]['body'], list):
            body = d['positive']['comments'][0]['body'][0].strip()
        else:
            body = d['positive']['comments'][0]['body'].strip()
        op = d['op_text'].split("EDIT:")[0].strip()
        datapoint = f"{random.choice(instruction_bank)}\n\nArgument: {op}\n\nCounter-argument: {body}"
        write_json_line(train_f, datapoint, "en", "https://chenhaot.com/pages/changemyview.html")

print("############################")
print("########## Lbox ###########")
print("############################")
source = "https://github.com/lbox-kr/lbox-open"

# statutes classification task
data_st_plus = load_dataset("lbox/lbox_open", "statute_classification_plus")
instruction_bank = ["For the given case facts predict the related South Korean legal statute.",
                    "When presented with this fact pattern what are the relevant legal statutes in South Korean law?"]
for x in data_st_plus["train"]:
    datapoint = f"{random.choice(instruction_bank)}\n\nFacts: {x['facts']}\nStatute(s):{','.join(x['statutes'])}"
    write_json_line(train_f, datapoint, "ko", source)

# Legal judgement prediction tasks
data_ljp_criminal = load_dataset("lbox/lbox_open", "ljp_criminal")
instruction_bank = [
    "Given these facts from a South Korean criminal law case. Predict the court's ruling and the reason for the ruling."]
for x in data_ljp_criminal["train"]:
    reason = ""
    if x["reason"] != "" and x["reason"] != -1:
        reason = f"Reason: {x['reason']}"
    datapoint = f"{random.choice(instruction_bank)}\n\nFacts: {x['facts']}\n{reason}\nRuling: {x['ruling']['text']}"
    write_json_line(train_f, datapoint, "ko", source)

data_ljp_civil = load_dataset("lbox/lbox_open", "ljp_civil")
for x in data_ljp_civil["train"]:
    datapoint = f"{random.choice(instruction_bank)}\n\nFacts: {x['facts'].strip()}\n\nClaims: {x['gist_of_claim']['text'].strip()}\n\nRuling: {x['ruling']['text']}"
    write_json_line(train_f, datapoint, "ko", source)

print("############################")
print("########## Sara ###########")
print("############################")
source = "https://arxiv.org/abs/2005.05257"
df = pd.read_csv("raw_data/sara.tsv", sep="\t", header=None)
entailment_instruction_bank = ["Consider the following US Tax scenario. Does the first fact entail the second fact?",
                               "Are these two sentences entailed or contradicting?",
                               "Respond entailement or contradiction to these two sentences."]
tax_liability_instruction_bank = ["Consider the following US Tax scenario and answer the question.",
                                  "Consider the following scenario. Calculate the right amount of tax liablity and answer the question."]
for i, row in df.iterrows():
    if "tail" in row[2] or "Contra" in row[2]:
        datapoint = f"{random.choice(entailment_instruction_bank)}\n\nSentence 1: {row[0]}\nSentence 2: {row[1]}\nAnswer: {row[2]}"
        write_json_line(train_f, datapoint, "en", source)
    else:
        datapoint = f"{random.choice(tax_liability_instruction_bank)}\n\nQuestion: {row[0]} {row[1]}\nAnswer: {row[2]}"
        write_json_line(train_f, datapoint, "en", source)

        value = int(row[2].replace("$", ""))
        options = [int(value + ((.5 - random.random()) * value)) for i in range(3)] + [value]
        random.shuffle(options)
        choices = ""
        for choice_value, option in zip(["(a)", "(b)", "(c)", "(d)"], options):
            choices += f"{choice_value} ${option}\n"
        correct = ["(a)", "(b)", "(c)", "(d)"][options.index(value)]
        datapoint = f"{random.choice(tax_liability_instruction_bank)} Denote your final answer with the \"Final Answer: The final answer is [CORRECT ANSWER]. I hope it is correct\".\n\nQuestion: {row[0]} {row[1]}\n{choices}\n\nFinal Answer: The final answer is {correct}. I hope it is correct."
        write_json_line(train_f, datapoint, "en", source)

print("############################")
print("########## BVA Decisions ###########")
print("############################")
json_files = [f"./raw_data/VetClaims-JSON/BVA Decisions JSON Format/{pos_json}" for pos_json in
              os.listdir("./raw_data/VetClaims-JSON/BVA Decisions JSON Format") if pos_json.endswith('.json')]
json_files.extend([f"./raw_data/VetClaims-JSON/BVA Decisions JSON Format +25/{pos_json}" for pos_json in
                   os.listdir("./raw_data/VetClaims-JSON/BVA Decisions JSON Format +25") if pos_json.endswith('.json')])
sentences = []
rule_trees = []
for json_f in json_files:
    with open(json_f, "r") as f:
        x = json.loads(f.read())
        sentences.extend(x["sentences"])
        rule_trees.append(x["ruleTree"])
instruction_bank = ["Label the sentence according to its rhetorical role in a legal argument.",
                    "Please label the sentence as according to its role as either a FindingSentence, a ReasoningSentence, a LegalRuleSentence, a CitationSentence, or an EvidenceSentence. If it is none of these, mark it as just Sentence.",
                    "Please label the following according to one of these categories.\n\tFindingSentence. A finding sentence is a sentence that primarily states an authoritative finding, conclusion or determination of the trier of fact – a decision made “as a matter of fact” instead of \"as a matter of law.\"\n\tReasoningSentence. A reasoning sentence is a sentence that primarily reports the trier of fact’s reasoning based on the evidence, or evaluation of the probative value of the evidence, in making the findings of fact.\n\tEvidenceSentence. An evidence sentence is a sentence that primarily states the content of the testimony of a witness, states the content of documents introduced into evidence, or describes other evidence.\n\tLegalRuleSentence. A legal-rule sentence is a sentence that primarily states one or more legal rules in the abstract, without stating whether the conditions of the rule(s) are satisfied in the case being decided.\n\tCitationSentence. A citation sentence is a sentence whose primary function is to reference legal authorities or other materials, and which usually contains standard notation that encodes useful information about the cited source.\n\tSentence. All other sentences."]


def turn_rule_tree_to_text(tree, n=0):
    op = ""
    if 'operation' in tree:
        op = tree['operation']
    elif 'inferenceRelation' in tree:
        op = tree['inferenceRelation']

    if "label" in tree:
        text = f"{op} {tree['label']}"
    else:
        text = f"{op} {tree['name']}"

    if "nodes" not in tree and "children" not in tree:
        return text
    else:
        if "nodes" in tree:
            children = tree["nodes"]
        else:
            children = tree["children"]
        if isinstance(children, dict):
            node_text = f"{''.join(['    '] * n)}{children['inferenceRelation']} {children['name']}"
        else:
            node_text = "\n".join(
                [f"{''.join(['    '] * n)}{turn_rule_tree_to_text(node, n + 1)}" for node in children])
        return f"{text}\n{node_text}"


for sentence in sentences:
    if 'rhetClass' in sentence:
        role = sentence['rhetClass']
    else:
        role = ",".join(sentence['rhetRole'])
    datapoint = f"{random.choice(instruction_bank)}\n\nSentence: {sentence['text'].strip()}\nRhetorical Role: {role.strip()}"
    write_json_line(train_f, datapoint, "en", "https://github.com/LLTLab/VetClaims-JSON")

instruction_bank = [
    "Take the following sentence, name all the rules that would be required to back up the claim. Do so in tree format with logical operators like AND and OR.",
    "Name all the rules that would be required to back up the claim."]
known_data = []
for tree_rule in rule_trees:
    tree_rule = turn_rule_tree_to_text(tree_rule)
    datapoint = f"{random.choice(instruction_bank)}\n\nClaim: {tree_rule.strip()}"
    if datapoint not in known_data:
        write_json_line(train_f, datapoint, "en", "https://github.com/LLTLab/VetClaims-JSON")
        known_data.append(datapoint)

### Reclor has logical reasoning.
print("############################")
print("########## Reclor ###########")
print("############################")
instruction_bank = ["Given the context answer the reasoning question.",
                    "Answer the logical reasoning multiple choice questions.",
                    "State the answer in the following format, \"Final Answer: The final answer is ([ANSWER]). I hope it is correct.\"",
                    "Read the passage any any relevant rules describing the world. Apply the rules to the facts to answer the question."]
with open("./raw_data/reclor_train.json", "r") as f:
    df = json.loads(f.read())
for data in df:
    options = ""
    options_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for x, lab in zip(data["answers"], options_labels):
        options += f"{lab} {x}\n"
    correct_option = options_labels[data['label']]
    datapoint = f"{random.choice(instruction_bank)}\n\nQuestion: {data['context']} {data['question']}\n{options}\nFinal Answer: The final answer is: {correct_option}. I hope it is correct."
    write_json_line(train_f, datapoint, "en", "https://github.com/yuweihao/reclor")

### CAIL 2022: https://github.com/china-ai-law-challenge/CAIL2022/tree/main/lblj/data/stage_2
print("############################")
print("########## CAIL2022 ###########")
print("############################")

with open("raw_data/cail2022_train_entry_lblj.jsonl", "r", encoding="utf8") as f:
    questions = [json.loads(x) for x in f.readlines()]

instruction_bank_mc = [
    "Use Chinese law to answer these multiple choice questions. Pick the best counter-argument to the plaintiff's argument.",
    "Which of these is the best response to the following argument if you were the defendant? Consider Chinese law."]
instruction_bank = ["Use Chinese law. What is the counter-argument to the plaintiff's argument?",
                    "How should Defendant respond to the following argument? Use Chinese law."]
instruction_bank_crime = ["Consider Chinese law, what is the likely crime being discussed here."]
lookup = ["(a)", "(b)", "(c)", "(d)", "(e)"]
source = "https://github.com/china-ai-law-challenge/CAIL2022/tree/main/lblj/data/stage_2"
for question in questions:
    datapoint = f"{random.choice(instruction_bank_mc)}\n\nPlaintiff's Argument:{question['sc']}\n\n(a) {question['bc_1']}\n(b) {question['bc_2']}\n(c) {question['bc_3']}\n(d) {question['bc_4']}\n(e) {question['bc_5']}"
    datapoint += "Best counter-argument: {lookup[question['answer'] - 1]}"
    write_json_line(train_f, datapoint, "zh", source)

    response = question[f"bc_{question['answer']}"]
    datapoint = f"{random.choice(instruction_bank)}\n\nPlaintiff's Argument:{question['sc']}\nDefendant's Response: {response}"
    write_json_line(train_f, datapoint, "zh", source)

    datapoint = f"{random.choice(instruction_bank_crime)}\n\nPlaintiff's Argument:{question['sc']}\nDefendant's Response: {response}\nCrime: {question['crime']}"
    write_json_line(train_f, datapoint, "zh", source)

    datapoint = f"{random.choice(instruction_bank_crime)}\n\n{question['sc']}\nCrime: {question['crime']}"
    write_json_line(train_f, datapoint, "zh", source)

print("############################")
print("########## CAIL2019 ###########")
print("############################")
instruction_bank = [
    "Consider the following passage from a Chinese legal case. Answer the questions about the case. If you cannot answer the question feel free to say as such.",
    "Consider the following situation in Chinese law, answer the questions. If the information is not in the passage, respond with, \"Sorry, this question cannot be answered based on the information available.\"",
    "Consider the following passage from a Chinese legal case. Answer the questions about the case. If the question is impossible to answer, say that it cannot be answered."]
with open("./raw_data/big_train_data.json", "r") as f:
    data = json.loads(f.read())["data"]
    for d in data:
        for paragraph in d['paragraphs']:
            for question in paragraph['qas']:
                if question['is_impossible']:
                    answer = "Sorry, this question cannot be answered based on the information available."
                else:
                    answer = ", ".join([a['text'] for a in question['answers']])
                datapoint = f"{random.choice(instruction_bank)}\n\n{paragraph['context']}\n\nQuestion:{question['question']}\nAnswer:{answer}"
                write_json_line(train_f, datapoint, "zh", "https://github.com/china-ai-law-challenge/CAIL2019")

print("############################")
print("########## Brazilian Bar Exam ###########")
print("############################")
with open("./raw_data/oab.json", "r") as f:
    qs = json.loads(f.read())

instruction_bank = ["Answer the questions from the Brazilian bar exam.",
                    "Answer these legal multiple choice questions according to Brazilian law."]


def all_law_articles_in_path(laws_path):
    # reads all .xml files in laws_path to a list of law_articles
    assert os.path.isdir(laws_path)
    laws = {}

    filelist = glob.glob(os.path.join(laws_path, "**/*.xml"), recursive=True)
    print(filelist)

    for file in filelist:
        urn, law = law_articles_in_file(file)
        laws[urn] = law
    return laws


def namespace_it(namespace, key, element):
    # namespaced element in {namespace}element syntax
    return "{{{}}}{}".format(namespace[key], element)


def lazy_articles_in_tree(tree):
    for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Artigo')):
        yield artigo.get('id'), ''.join(artigo.itertext())
    for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Caput')):
        yield artigo.get('id'), ''.join(artigo.itertext())
    for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Paragrafo')):
        yield artigo.get('id'), ''.join(artigo.itertext())
    for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Inciso')):
        yield artigo.get('id'), ''.join(artigo.itertext())
    for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Alinea')):
        yield artigo.get('id'), ''.join(artigo.itertext())


def articles_in_tree(tree):
    return list(lazy_articles_in_tree(tree))


def law_articles_in_file(law_path):
    law_xml = parse_xml(law_path)
    law_urn = get_urn(law_xml)
    return (law_urn, articles_in_tree(law_xml))


def elements_in_tree(tree, element_tag):
    assert isinstance(tree, etree._ElementTree)
    for element in tree.getiterator(element_tag):
        yield element


def parse_xml(path, parser=etree.XMLParser(remove_blank_text=True)):
    return etree.parse(path)


def get_urn(law_xml):
    assert isinstance(law_xml, etree._ElementTree)
    # fixme http://lxml.de/xpathxslt.html#namespaces-and-prefixes
    id_element = law_xml.find(
        namespace_it(law_xml.getroot().nsmap, None, 'Metadado') + '/' + namespace_it(law_xml.getroot().nsmap, None,
                                                                                     'Identificacao'))
    return id_element.get('URN')


leis = all_law_articles_in_path('./raw_data/oab_lexml/')
import yaml

justifications = []
with open("./raw_data/oab_ethics.yaml", "r") as stream:
    try:
        justifications.extend(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

with open("./raw_data/oab_const.yaml", "r") as stream:
    try:
        justifications.extend(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
just_dict = defaultdict(dict)
for just in justifications:
    just_dict[just["exam"]][str(just["question"])] = just

for q in qs:
    choices = ""
    correct_answer = None
    if not q["valid"]:
        continue
    for c in q["options"]:
        choices += f"({c['letter'].lower()}) {c['text']}\n"
        if c["correct"]:
            correct_answer = f"({c['letter'].lower()})"
    if correct_answer is not None:
        datapoint = f"{random.choice(instruction_bank)}\n\nQuestion: {q['enum']}\n{choices}"

        legal_text = None
        if q["filename"].split(".txt")[0] in just_dict and q["number"] in just_dict[q["filename"].split(".txt")[0]]:
            if isinstance(just_dict[q["filename"].split(".txt")[0]][q["number"]]["urn"], str):
                urns = [just_dict[q["filename"].split(".txt")[0]][q["number"]]["urn"]]
            legal_texts = []
            for urn in urns[:1]:
                law = leis[urn.split("!")[0]]

                for article, text in law:
                    if article == urn.split("!")[1]:
                        legal_texts.append(text.strip().replace("\n", ""))
            legal_text = "\n".join(legal_texts)
            if legal_text is not None and legal_text.strip() != "":
                datapoint += f"\n\nRule(s): {legal_text}"
            if "comment" in just_dict[q["filename"].split(".txt")[0]][q["number"]] and \
                    just_dict[q["filename"].split(".txt")[0]][q["number"]]["comment"] is not None:
                analysis = just_dict[q["filename"].split(".txt")[0]][q["number"]]["comment"].replace("\n", "")
                datapoint += f'\n\nAnalysis: {analysis}'
        datapoint += f"\nAnswer: {correct_answer}."
        write_json_line(train_f, datapoint, "pt", "https://arxiv.org/pdf/1712.05128.pdf")

# Legal Stack Exchange questions are usually high quality
print("############################")
print("########## Stack Exchange Questions (LEGAL) ###########")
print("############################")

df = pd.read_csv("./raw_data/stack-exchange.csv")
instruction_bank = [
    "Answer the following legal question. Cite relevant evidence when possible.",
    "Answer this online forum question about the law.",
    "Provide an explanation for this short form legal question."
]
n_examples = 3
curr_examples = 0
for idx, example in df.iterrows():
    soup = BeautifulSoup(example["body"])
    text = soup.get_text()
    question = text
    soup = BeautifulSoup(example["body.1"])
    text = soup.get_text()
    answer = text
    instruction = f"{random.choice(instruction_bank)}"
    if random.random() > .7:
        instruction += " " + f"This question is about: {','.join([x.replace('>', '').replace('<', '').replace('-', ' ').strip() for x in example['tags'].split('>') if x.replace('>', '').replace('<', '').strip() != ''])}."

    datapoint = f"{instruction}\n\nQuestion: {question}\nAnswer: {answer}"
    write_json_line(train_f, datapoint, "en", "https://law.stackexchange.com/")

print("############################")
print("##########  LegalQA ZHO ###########")
print("############################")
df = pd.read_csv("./raw_data/LegalQA-all-train.csv")

df = df[df['label'] == 1]

instruction_bank = [
    "Answer the following question according to Chinese law, use plain language as if you are a lawyer answering on an online forum.",
    "This is a question on a Chinese online forum for legal advice. Do not cite case law and use plain language.",
    "Answer the question as a lawyer according to Chinese law, be informal."]
for q, a in zip(df['question: body'], df['answer']):
    datapoint = f"{random.choice(instruction_bank)}\n\nQ:{q}\nA:{a}"
    write_json_line(train_f, datapoint, "zh", "https://github.com/siatnlp/LegalQA")

print("############################")
print("########## Privacy QA ###########")
print("############################")
df = pd.read_csv("./raw_data/policy_train_data.csv", sep="\t")

for index, example in df.iterrows():
    datapoint = f"Determine if the term mentioned from the privacy policy is relevant or irrelevant to the given question.\n\nQ: {example['Query']}\nTerm: {example['Segment']}\nA: {example['Label']}"
    write_json_line(train_f, datapoint, "en", "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP")

# Thai supreme court case law
print("############################")
print("########## TSCC ALQAC ###########")
print("############################")
with open("./raw_data/tscc_alqac2021_question.train.json", "r") as f:
    cases = json.loads(f.read())

with open("./raw_data/tscc_alqac2021_law.json", "r") as f:
    laws = json.loads(f.read())

laws_dict = {}
for article in laws[0]['articles']:
    laws_dict[article['id']] = article['text']

instructions_bank = [
    "For the relevant facts, please provide the relevant Thai law(s). Use the rule to determine the court's likely conclusion.",
    "Given these facts in the Thai legal system, please output the relevant legal rule(s) and the court's likely judgement.",
    "Given these facts in the Thai legal system, please output the relevant legal rule(s) and provide the legal conclusion of whether the court is likely to find for or against the defendant.",
    "Given these facts in the Thai legal system, please output the relevant legal rule(s) and provide the legal conclusion of whether the court is likely to find the defendant guilty or not guilty.",
]

source = "https://github.com/KevinMercury/tscc-dataset-alqac2021/blob/main/tscc_alqac2021_law.json"
for case in cases:
    text = case["text"]
    relevant_articles = []
    for article in case["relevant_articles"]:
        law_text = laws_dict[article['article_id']]
        relevant_articles.append(law_text)

    # Provide a MC version for the judgement
    if random.random() > .5:
        outcome = "The court would likely find the defendant guilty." if case[
                                                                             "label"] == 1 else "The court would likely find the defendant not guilty."
    else:
        outcome = "The court would rule against the defendant." if case[
                                                                       "label"] == 1 else "The court would rule for the defendant."
    laws = '\n'.join(relevant_articles)
    datapoint = f"{random.choice(instructions_bank)}\n\nFacts: {text}\nLaw(s): {laws}\nConclusion: {outcome}"
    write_json_line(train_f, datapoint, "th", source)

    # Provide a non-MC version
    outcome_mc1 = ["(a)", "(b)"][case["label"]]
    datapoint = f"{random.choice(instructions_bank)}\n\nQuestion: {text} How would the court find?\n(a) For the defendant.\n(b) Against the defendant.\nLaw(s): {laws}\nAnswer: {outcome_mc1}."
    write_json_line(train_f, datapoint, "th", source)

    outcome_mc1 = ["(b)", "(a)"][case["label"]]
    datapoint = f"{random.choice(instructions_bank)}\n\nQuestion: {text} How would the court find?\n(a) Against the defendant.\n(b) For the defendant.\nLaw(s): {laws}\nAnswer: {outcome_mc1}."
    write_json_line(train_f, datapoint, "th", source)

# Case briefs take the form of a question and an answer.
print("############################")
print("########## CaseBriefs ###########")
print("############################")
case_brief_instructions = [
    "Given the key facts of a case, provide the core question the court should answer, then provide an analysis for how the a court might decide the case.",
    "Given the facts, describe how the court should think about the key issue?"]

df = load_dataset("socratic-machines/case-briefs", "combined", use_auth_token=True)

for example in df["train"]["text"]:
    example = example.split("Key Facts:")[0].split("Year:")[0]
    example = example.replace("Answer:", "Analysis:")
    example = f"{random.choice(case_brief_instructions)}\n\n{example}"
    write_json_line(train_f, example, "en", "https://www.oyez.org")

# OLC memos start off with a short form summary and then write the memo
print("############################")
print("########## OLC Memos with instruction ###########")
print("############################")
df = load_dataset("pile-of-law/pile-of-law", "olc_memos", split="train")

instruction_bank = ["Write a legal research memo on the following topic.",
                    "Write a memo in the style of OLC on the following legal research question.",
                    "Write a memo in the form of U.S. Office of Legal Counsel.",
                    "Consider the question below, write a formal legal research memo."]

for example in df["text"]:
    if example.startswith("b'"):
        example = example.encode().decode('unicode-escape').encode('latin1').decode('utf-8')[2:-2].strip()
    datapoint = f"{random.choice(instruction_bank)}\n\n{example}"
    write_json_line(train_f, datapoint, "en", "pile-of-law/pile-of-law/olc_memos")

print("############################")
print("########## Reddit Legal QA ###########")
print("############################")
reddit_instructions = [
    "Here is someone's legal concern. Answer as if you were replying on Reddit. If you are not a lawyer, include the disclaimer IANAL.",
    "Here is someone's legal question. Advice them on the situation. Think like a lawyer on Reddit."]

df = load_dataset("pile-of-law/pile-of-law", "r_legaladvice", split="train")

for example in df["text"]:
    question = example.split("Question:")[-1]
    q = question.split("Answer #")[0]
    if "deleted" in example.lower() or "removed" in example.lower():
        continue
    answers = question.split("Answer #")[1:]
    answers = [a.split(":")[-1] for a in answers]
    for a in answers:
        datapoint = f"Question: {q}\n\nAnalysis: {a}"
        write_json_line(train_f, datapoint, "en", "pile-of-law/pile-of-law/r_legaladvice")


train_f.close()
