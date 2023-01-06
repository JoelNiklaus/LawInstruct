
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

from abstract_dataset import write_json_line, MAX_FILE_SIZE, get_output_file_name, TASK_TYPE, JURISDICTION

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

# TODO maybe later save instructions and answers into different columns for MT

# TODO translate instruction banks into other EU languages


# TODO put the openly available datasets on huggingface and then use the datasets library to load them

### Reclor has logical reasoning.
print("############################")
print("########## Reclor ###########")
print("############################")
instruction_bank = ["Given the context answer the reasoning question.",
                    "Answer the logical reasoning multiple choice questions.",
                    "State the answer in the following format, \"Final Answer: The final answer is ([ANSWER]). I hope it is correct.\"",
                    "Read the passage any any relevant rules describing the world. Apply the rules to the facts to answer the question."]
source = "https://github.com/yuweihao/reclor"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.N_A

with open("./raw_data/reclor_train.json", "r") as f:
    df = json.loads(f.read())
for data in df:
    options = ""
    options_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for x, lab in zip(data["answers"], options_labels):
        options += f"{lab} {x}\n"
    correct_option = options_labels[data['label']]
    datapoint = f"{random.choice(instruction_bank)}\n\nQuestion: {data['context']} {data['question']}\n{options}\nFinal Answer: The final answer is: {correct_option}. I hope it is correct."
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

# Add math-type reasoning b/c tax has that flavor
print("############################")
print("########## gsm8k ###########")
print("############################")
source = "https://huggingface.co/datasets/gsm8k"
x = load_dataset("gsm8k", "main", split="train")
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.N_A

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
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

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
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

print("############################")
print("########## Lila ###########")
print("############################")
json_files = [pos_json for pos_json in os.listdir("raw_data/all_lila/") if pos_json.endswith('.json')]
instruction_bank = ["Consider the following question. Write a Python program to solve it.",
                    "Write a Python program to solve the following question, denote it as \"Program:\". Provide the output as \"Answer:\"."]
source = "https://github.com/allenai/Lila"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.N_A

for json_file in json_files:
    with open(os.path.join("raw_data/all_lila/", json_file), "r") as f:
        loaded_file = json.loads(f.read())
        for example in loaded_file["Instances"]:
            if example["split"] != "train":
                continue
            for program, answer in zip(example['Output Program'], example['Output Answer']):
                datapoint = f"{random.choice(instruction_bank)}\n\nQuestion: {example['Input']}\nProgram:\n```python\n{program}\n```\nAnswer: {answer}"
                write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

print("############################")
print("########## Sara Prolog ###########")
print("############################")
# TODO do we have an url here?
source = "sara"
instruction_bank = ["Convert the following statute into prolog code.",
                    "Write a prolog program to convert the statute into code, denote it as \"Prolog Program:\"."]
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.US

json_files = [pos_json for pos_json in os.listdir("raw_data/sara_statutes/source")]
for json_file in json_files:
    with open(os.path.join("raw_data/sara_statutes/source/", json_file), "r") as f_normal:
        with open(os.path.join("raw_data/sara_statutes/prolog/", json_file) + ".pl", "r") as f_prolog:
            datapoint = f"{random.choice(instruction_bank)}\n\nStatute:\n{f_normal.read()}\n\nProlog Program:\n\n{f_prolog.read()}"
            write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

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
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

print("############################")
print("########## Sara ###########")
print("############################")
df = pd.read_csv("raw_data/sara.tsv", sep="\t", header=None)
source = "https://arxiv.org/abs/2005.05257"
jurisdiction = JURISDICTION.US
entailment_instruction_bank = ["Consider the following US Tax scenario. Does the first fact entail the second fact?",
                               "Are these two sentences entailed or contradicting?",
                               "Respond entailement or contradiction to these two sentences."]
tax_liability_instruction_bank = ["Consider the following US Tax scenario and answer the question.",
                                  "Consider the following scenario. Calculate the right amount of tax liablity and answer the question."]
for i, row in df.iterrows():
    if "tail" in row[2] or "Contra" in row[2]:
        task_type = TASK_TYPE.NATURAL_LANGUAGE_INFERENCE
        datapoint = f"{random.choice(entailment_instruction_bank)}\n\nSentence 1: {row[0]}\nSentence 2: {row[1]}\nAnswer: {row[2]}"
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)
    else:
        task_type = TASK_TYPE.QUESTION_ANSWERING
        datapoint = f"{random.choice(tax_liability_instruction_bank)}\n\nQuestion: {row[0]} {row[1]}\nAnswer: {row[2]}"
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

        value = int(row[2].replace("$", ""))
        options = [int(value + ((.5 - random.random()) * value)) for i in range(3)] + [value]
        random.shuffle(options)
        choices = ""
        for choice_value, option in zip(["(a)", "(b)", "(c)", "(d)"], options):
            choices += f"{choice_value} ${option}\n"
        correct = ["(a)", "(b)", "(c)", "(d)"][options.index(value)]
        datapoint = f"{random.choice(tax_liability_instruction_bank)} Denote your final answer with the \"Final Answer: The final answer is [CORRECT ANSWER]. I hope it is correct\".\n\nQuestion: {row[0]} {row[1]}\n{choices}\n\nFinal Answer: The final answer is {correct}. I hope it is correct."
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

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
source = "civpro_questions"  # TODO do we have an url here?
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.US

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
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

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
df = load_dataset("hendrycks_test", "professional_law", split="auxiliary_train").select(range(1200))
task_type = TASK_TYPE.MULTIPE_CHOICE
jurisdiction = JURISDICTION.US


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
    write_json_line(train_f, final_datapoint, "en", source, task_type, jurisdiction)

    datapoint_zero_shot = datapoint.replace("The Final Answer: ", "Answer: ").split("###")[-1].strip()
    final_datapoint_zero_shot = random.choice(instructions_zero_shot) + "\n\n" + datapoint_zero_shot
    write_json_line(train_f, final_datapoint_zero_shot, "en", source, task_type, jurisdiction)

print("############################")
print("########## MBE ###########")
print("############################")

df = pd.read_csv("raw_data/mbe_train.csv")
source = "MBE"  # TODO do we have an url for the source here?
task_type = TASK_TYPE.MULTIPE_CHOICE
jurisdiction = JURISDICTION.US

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
    write_json_line(train_f, final_datapoint, "en", source, task_type, jurisdiction)

    if isinstance(subject, str) and subject.strip() != "":
        datapoint = f"{random.choice(instruction_bank_subject)}\n\n{data_no_answer}\nSubject: {subject}"

        datapoint = random.choice(instructions_examples) + "\n\n" + datapoint
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

        datapoint = random.choice(instruction_bank_subject_generation) + subject + ".\n\n" + datapoint_with_answer
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

print("############################")
print("########## Littleton ###########")
print("############################")
json_files = [pos_json for pos_json in os.listdir("raw_data/littleton/examples/") if pos_json.endswith('.json')]
instruction_bank = [
    "Consider the law of future interests and conveyances in American property law. Consider the chain of events and then state the interests.",
    "According to American law, consider the chain of events and future interests."]
source = "https://github.com/grimmelm/littleton"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.US

for json_file in json_files:
    with open(os.path.join("raw_data/littleton/examples/", json_file), "r") as f:
        loaded_file = json.loads(f.read())[1]
        if isinstance(loaded_file, str):
            continue
        for example in loaded_file["examples"]:
            datapoint = f"{random.choice(instruction_bank)}\n\nEvents: {example['program']}\nAnswer: {example['result']}"
            write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

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
            write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

print("############################")
print("########## JEC-QA ###########")
print("############################")
instruction_bank = [
    "Answer these multiple choice reasoning questions about Chinese Law. Select all answers that apply, you may have multiple correct answers.",
    "Answer these Chinese Law multiple choice questions, you might have multiple correct answers. Denote your answer(s) as \"Answer: [answer(s)].\""]
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.CHINA

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


print("############################")
print("########## ILDC Dataset ###########")
print("############################")

df1 = pd.read_csv("raw_data/ILDC_multi.csv")
df1 = df1[df1["split"] == "train"]
df2 = pd.read_csv("raw_data/ILDC_single.csv")
df2 = df2[df2["split"] == "train"]

source = "https://github.com/Exploration-Lab/CJPE"
task_type = TASK_TYPE.TEXT_CLASSIFICATION
jurisdiction = JURISDICTION.INDIA
instruction_bank = [
    "According to Indian law, will this petition be accepted? If there is more than one petition consider whether the court will accept at least one.",
    "Will the court accept or reject this petition? Use Indian law. If there is more than one petition consider whether the court will accept at least one."]

for idx, row in df1.iterrows():
    decision = "Court Decision: Reject" if row["label"] == 0 else "Court Decision: Accept"
    datapoint = f"{random.choice(instruction_bank)}\n\n{row['text']}\n\n{decision}"
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

for idx, row in df2.iterrows():
    decision = "Court Decision: Reject" if row["label"] == 0 else "Court Decision: Accept"
    datapoint = f"{random.choice(instruction_bank)}\n\n{row['text']}\n\n{decision}"
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

# Scraped bar exam essays
print("############################")
print("########## CA Bar Exam Essays ###########")
print("############################")

source = "https://www.calbar.ca.gov/Admissions/Examinations/California-Bar-Examination/Past-Exams"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.US

with open("raw_data/bar_exam_essays_ca.jsonl") as f:
    exams = [json.loads(x) for x in f.readlines()]
    for exam in exams:
        write_json_line(train_f, exam['text'], "en", source, task_type, jurisdiction)

print("############################")
print("########## MC Exams Law ###########")
print("############################")

df = pd.read_csv("raw_data/raw_legal_mc_with_explanations.csv")
task_type = TASK_TYPE.MULTIPLE_CHOICE
jurisdiction = JURISDICTION.US

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
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

    # Chain of thought
    datapoint = f"{random.choice(instruction_bank_expl)}\n\nQ:{q}\nExplanation: {explanation}\nA:{a}"
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

print("############################")
print("########## Korean LegalQA ###########")
print("############################")
source = "https://raw.githubusercontent.com/haven-jeon/LegalQA/main/data/legalqa.jsonlines"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.SOUTH_KOREA

instruction_bank = ["Consider the following question. Retrieve the relevant South Korean legal article.",
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
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.SPAIN

instruction_bank = [
    "Consider this Spanish Labor Law translated passage. Answer the question using an extractive snippet of text.",
    "Consider this Spanish Labor Law translated passage. Answer the question from the context.",
    "Answer the following Spanish labor law question given the legal provision."]
for idx, row in df.iterrows():
    question, context, answer = row["Question"], row["context"], row["Answer text"]
    datapoint = f"{random.choice(instruction_bank)}\n\nContext: {context}\nQ: {question}\nA: {answer}"
    write_json_line(train_f, datapoint, "es", source, task_type, jurisdiction)

# International citizenship law questions
print("############################")
print("########## International citizenship law questions ###########")
print("############################")
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.INTERNATIONAL
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

    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

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
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

# Will Validity
print("############################")
print("########## Valid Wills ###########")
print("############################")
train = pd.read_csv('./raw_data/wills_train.csv', encoding='utf-8')  # replace with real path and dataset names
instruction_bank = [
    "Given a statement in a will, the relevant U.S. law, is the condition supported, refuted, or unrelated.",
    "Is the statement in the will valid given the law and conditions? Answer with one of unrelated, supported, refuted."]
source = "https://arxiv.org/pdf/2210.16989.pdf"
task_type = TASK_TYPE.TEXT_CLASSIFICATION
jurisdiction = JURISDICTION.US

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
    write_json_line(train_f, prompt, "en", source, task_type, jurisdiction)
    write_json_line(train_f, prompt2, "en", source, task_type, jurisdiction)
    write_json_line(train_f, prompt_mc, "en", source, task_type, jurisdiction)

# Chinese Bar Exam, no explanations.
print("############################")
print("########## LogiQA ###########")
print("############################")
instruction_bank = [
    "Answer these multiple choice reasoning questions about Chinese Law. There is only one right answer.",
    "Answer these Chinese Law multiple choice questions. There is only one correct answer. Denote your answer as \"Answer: [answer].\""]
source = "https://github.com/lgw863/LogiQA-dataset"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.CHINA

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
        write_json_line(train_f, datapoint, "zh", source, task_type, jurisdiction)
        if i >= len(x): break

# ChangeMyView Argumentation
print("############################")
print("########## ChangeMyView ###########")
print("############################")
instruction_bank = ["You are given a position, create an argument that would change the original poster's mind.",
                    "Write a counter argument to the proposal.", "Write a counter argument to the r/changemyview post.",
                    "Write a counterargument to this reddit post."]
source = "https://chenhaot.com/pages/changemyview.html"
task_type = TASK_TYPE.ARGUMENTATION
jurisdiction = JURISDICTION.UNKNOWN

with open("./raw_data/train_pair_data.jsonlist") as f:
    x = [json.loads(s) for s in f.readlines()]
    for d in x:
        if isinstance(d['positive']['comments'][0]['body'], list):
            body = d['positive']['comments'][0]['body'][0].strip()
        else:
            body = d['positive']['comments'][0]['body'].strip()
        op = d['op_text'].split("EDIT:")[0].strip()
        datapoint = f"{random.choice(instruction_bank)}\n\nArgument: {op}\n\nCounter-argument: {body}"
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

# statutes classification task
print("############################")
print("########## Lbox ###########")
print("############################")
data_st_plus = load_dataset("lbox/lbox_open", "statute_classification_plus")
source = "https://github.com/lbox-kr/lbox-open"
task_type = TASK_TYPE.TEXT_CLASSIFICATION
jurisdiction = JURISDICTION.SOUTH_KOREA
instruction_bank = ["For the given case facts predict the related South Korean legal statute.",
                    "When presented with this fact pattern what are the relevant legal statutes in South Korean law?"]
for x in data_st_plus["train"]:
    datapoint = f"{random.choice(instruction_bank)}\n\nFacts: {x['facts']}\nStatute(s):{','.join(x['statutes'])}"
    write_json_line(train_f, datapoint, "ko", source, task_type, jurisdiction)

# Legal judgement prediction tasks
data_ljp_criminal = load_dataset("lbox/lbox_open", "ljp_criminal")
instruction_bank = [
    "Given these facts from a South Korean criminal law case. Predict the court's ruling and the reason for the ruling."]
for x in data_ljp_criminal["train"]:
    reason = ""
    if x["reason"] != "" and x["reason"] != -1:
        reason = f"Reason: {x['reason']}"
    datapoint = f"{random.choice(instruction_bank)}\n\nFacts: {x['facts']}\n{reason}\nRuling: {x['ruling']['text']}"
    write_json_line(train_f, datapoint, "ko", source, task_type, jurisdiction)

data_ljp_civil = load_dataset("lbox/lbox_open", "ljp_civil")
for x in data_ljp_civil["train"]:
    datapoint = f"{random.choice(instruction_bank)}\n\nFacts: {x['facts'].strip()}\n\nClaims: {x['gist_of_claim']['text'].strip()}\n\nRuling: {x['ruling']['text']}"
    write_json_line(train_f, datapoint, "ko", source, task_type, jurisdiction)

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


source = "https://github.com/LLTLab/VetClaims-JSON"
task_type = TASK_TYPE.TEXT_CLASSIFICATION
jurisdiction = JURISDICTION.US

for sentence in sentences:
    if 'rhetClass' in sentence:
        role = sentence['rhetClass']
    else:
        role = ",".join(sentence['rhetRole'])
    datapoint = f"{random.choice(instruction_bank)}\n\nSentence: {sentence['text'].strip()}\nRhetorical Role: {role.strip()}"
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

task_type = TASK_TYPE.QUESTION_ANSWERING
instruction_bank = [
    "Take the following sentence, name all the rules that would be required to back up the claim. Do so in tree format with logical operators like AND and OR.",
    "Name all the rules that would be required to back up the claim."]
known_data = []
for tree_rule in rule_trees:
    tree_rule = turn_rule_tree_to_text(tree_rule)
    datapoint = f"{random.choice(instruction_bank)}\n\nClaim: {tree_rule.strip()}"
    if datapoint not in known_data:
        write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)
        known_data.append(datapoint)

### CAIL 2022: https://github.com/china-ai-law-challenge/CAIL2022/tree/main/lblj/data/stage_2
print("############################")
print("########## CAIL2022 ###########")
print("############################")
source = "https://github.com/china-ai-law-challenge/CAIL2022/tree/main/lblj/data/stage_2"
jurisdiction = JURISDICTION.CHINA

with open("raw_data/cail2022_train_entry_lblj.jsonl", "r", encoding="utf8") as f:
    questions = [json.loads(x) for x in f.readlines()]

instruction_bank_mc = [
    "Use Chinese law to answer these multiple choice questions. Pick the best counter-argument to the plaintiff's argument.",
    "Which of these is the best response to the following argument if you were the defendant? Consider Chinese law."]
instruction_bank = ["Use Chinese law. What is the counter-argument to the plaintiff's argument?",
                    "How should Defendant respond to the following argument? Use Chinese law."]
instruction_bank_crime = ["Consider Chinese law, what is the likely crime being discussed here."]
lookup = ["(a)", "(b)", "(c)", "(d)", "(e)"]
for question in questions:
    task_type = TASK_TYPE.MULTIPLE_CHOICE
    datapoint = f"{random.choice(instruction_bank_mc)}\n\nPlaintiff's Argument:{question['sc']}\n\n(a) {question['bc_1']}\n(b) {question['bc_2']}\n(c) {question['bc_3']}\n(d) {question['bc_4']}\n(e) {question['bc_5']}"
    datapoint += "Best counter-argument: {lookup[question['answer'] - 1]}"
    write_json_line(train_f, datapoint, "zh", source, task_type, jurisdiction)

    task_type = TASK_TYPE.QUESTION_ANSWERING
    response = question[f"bc_{question['answer']}"]
    datapoint = f"{random.choice(instruction_bank)}\n\nPlaintiff's Argument:{question['sc']}\nDefendant's Response: {response}"
    write_json_line(train_f, datapoint, "zh", source, task_type, jurisdiction)

    task_type = TASK_TYPE.TEXT_CLASSIFICATION
    datapoint = f"{random.choice(instruction_bank_crime)}\n\nPlaintiff's Argument:{question['sc']}\nDefendant's Response: {response}\nCrime: {question['crime']}"
    write_json_line(train_f, datapoint, "zh", source, task_type, jurisdiction)

    datapoint = f"{random.choice(instruction_bank_crime)}\n\n{question['sc']}\nCrime: {question['crime']}"
    write_json_line(train_f, datapoint, "zh", source, task_type, jurisdiction)

print("############################")
print("########## CAIL2019 ###########")
print("############################")
source = "https://github.com/china-ai-law-challenge/CAIL2019"
jurisdiction = JURISDICTION.CHINA

instruction_bank = [
    "Consider the following passage from a Chinese legal case. Answer the questions about the case. If you cannot answer the question feel free to say as such.",
    "Consider the following situation in Chinese law, answer the questions. If the information is not in the passage, respond with, \"Sorry, this question cannot be answered based on the information available.\"",
    "Consider the following passage from a Chinese legal case. Answer the questions about the case. If the question is impossible to answer, say that it cannot be answered."]
task_type = TASK_TYPE.QUESTION_ANSWERING
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
                write_json_line(train_f, datapoint, "zh", source, task_type, jurisdiction)

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

source = "https://arxiv.org/pdf/1712.05128.pdf"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.BRAZIL

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
        write_json_line(train_f, datapoint, "pt", source, task_type, jurisdiction)

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
source = "https://law.stackexchange.com/"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.UNKNOWN

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
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

print("############################")
print("##########  LegalQA ZHO ###########")
print("############################")
df = pd.read_csv("./raw_data/LegalQA-all-train.csv")

df = df[df['label'] == 1]

instruction_bank = [
    "Answer the following question according to Chinese law, use plain language as if you are a lawyer answering on an online forum.",
    "This is a question on a Chinese online forum for legal advice. Do not cite case law and use plain language.",
    "Answer the question as a lawyer according to Chinese law, be informal."]
source = "https://github.com/siatnlp/LegalQA"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.CHINA

for q, a in zip(df['question: body'], df['answer']):
    datapoint = f"{random.choice(instruction_bank)}\n\nQ:{q}\nA:{a}"
    write_json_line(train_f, datapoint, "zh", source, task_type, jurisdiction)

print("############################")
print("########## Privacy QA ###########")
print("############################")
df = pd.read_csv("./raw_data/policy_train_data.csv", sep="\t")
source = "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.UNKOWN

for index, example in df.iterrows():
    datapoint = f"Determine if the term mentioned from the privacy policy is relevant or irrelevant to the given question.\n\nQ: {example['Query']}\nTerm: {example['Segment']}\nA: {example['Label']}"
    write_json_line(train_f, datapoint, "en", source, task_type, jurisdiction)

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

given_facts_output_rules = "Given these facts in the Thai legal system, please output the relevant legal rule(s)"
instructions_bank = [
    "For the relevant facts, please provide the relevant Thai law(s). Use the rule to determine the court's likely conclusion.",
    f"{given_facts_output_rules} and the court's likely judgement.",
    f"{given_facts_output_rules} and provide the legal conclusion of whether the court is likely to find for or against the defendant.",
    f"{given_facts_output_rules} and provide the legal conclusion of whether the court is likely to find the defendant guilty or not guilty.",
]

source = "https://github.com/KevinMercury/tscc-dataset-alqac2021/blob/main/tscc_alqac2021_law.json"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.THAILAND

for case in cases:
    text = case["text"]
    relevant_articles = []
    for article in case["relevant_articles"]:
        law_text = laws_dict[article['article_id']]
        relevant_articles.append(law_text)

    # Provide a MC version for the judgement
    if random.random() > .5:
        outcome = f"The court would likely find the defendant{'' if case['label'] == 1 else ' not'} guilty."
    else:
        outcome = f"The court would rule {'against' if case['label'] == 1 else 'for'} the defendant."
    laws = '\n'.join(relevant_articles)
    datapoint = f"{random.choice(instructions_bank)}\n\nFacts: {text}\nLaw(s): {laws}\nConclusion: {outcome}"
    write_json_line(train_f, datapoint, "th", source, task_type, jurisdiction)

    # Provide a non-MC version
    outcome_mc1 = ["(a)", "(b)"][case["label"]]
    datapoint = f"{random.choice(instructions_bank)}\n\nQuestion: {text} How would the court find?\n(a) For the defendant.\n(b) Against the defendant.\nLaw(s): {laws}\nAnswer: {outcome_mc1}."
    write_json_line(train_f, datapoint, "th", source, task_type, jurisdiction)

    outcome_mc1 = ["(b)", "(a)"][case["label"]]
    datapoint = f"{random.choice(instructions_bank)}\n\nQuestion: {text} How would the court find?\n(a) Against the defendant.\n(b) For the defendant.\nLaw(s): {laws}\nAnswer: {outcome_mc1}."
    write_json_line(train_f, datapoint, "th", source, task_type, jurisdiction)

# Case briefs take the form of a question and an answer.
print("############################")
print("########## CaseBriefs ###########")
print("############################")
case_brief_instructions = [
    "Given the key facts of a case, provide the core question the court should answer, then provide an analysis for how the an American court might decide the case.",
    "Given the facts, describe how an American court should think about the key issue?"]

df = load_dataset("socratic-machines/case-briefs", "combined", use_auth_token=True)
source = "https://www.oyez.org"
task_type = TASK_TYPE.QUESTION_ANSWERING
jurisdiction = JURISDICTION.US

for example in df["train"]["text"]:
    example = example.split("Key Facts:")[0].split("Year:")[0]
    example = example.replace("Answer:", "Analysis:")
    example = f"{random.choice(case_brief_instructions)}\n\n{example}"
    write_json_line(train_f, example, "en", source, task_type, jurisdiction)


train_f.close()
