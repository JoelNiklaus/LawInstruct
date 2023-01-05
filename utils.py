import enum
import json
import datetime

MAX_FILE_SIZE = 6.25e8

TASK_TYPE = enum.Enum('TASK_TYPE', [
    # TODO is this detailed enough or do we need to distinguish topic classification from judgment prediction or NER from argument mining?
    'TEXT_CLASSIFICATION',
    'QUESTION_ANSWERING',
    'SUMMARIZATION',
    'NAMED_ENTITY_RECOGNITION',
    'NATURAL_LANGUAGE_INFERENCE',
    'MULTIPLE_CHOICE',
    'ARGUMENTATION'
    'UNKNOWN'
])

JURISDICTION = enum.Enum('JURISDICTION', [
    # EU
    'AUSTRIA', 'BELGIUM', 'BULGARIA', 'CROATIA', 'CZECHIA', 'DENMARK', 'ESTONIA', 'FINLAND',
    'FRANCE', 'GERMANY', 'GREECE', 'HUNGARY', 'IRELAND', 'ITALY', 'LATVIA', 'LITHUANIA', 'LUXEMBOURG',
    'MALTA', 'NETHERLANDS', 'POLAND', 'PORTUGAL', 'ROMANIA', 'SLOVAKIA', 'SLOVENIA', 'SPAIN', 'SWEDEN',
    # Europa
    'EU', 'SWITZERLAND', 'UK',
    # Asia
    'CHINA', 'INDIA', 'JAPAN', 'SOUTH_KOREA', 'THAILAND',
    # North America
    'US', 'CANADA',
    # South America
    'BRAZIL',
    'INTERNATIONAL',  # international law
    'UNKNOWN',  # we don't know the jurisdiction
    'N_A'  # Not a legal task
])


def write_json_line(file, text: str, lang: str, source: str,
                    task_type: TASK_TYPE = TASK_TYPE.UNKNOWN,
                    jurisdiction: JURISDICTION = JURISDICTION.UNKNOWN):
    file.write(json.dumps({
        "lang": lang,
        "jurisdiction": jurisdiction,
        "task_type": task_type,
        "source": source,
        "text": text,
        "downloaded_timestamp": datetime.date.today().strftime("%m-%d-%Y")
    }) + "\n")


def get_output_file_name(dataset_name, file_idx=0, split='train'):
    # we save each dataset to a separate file so we only need to generate new datasets
    return f"./data/{dataset_name}.{split}.{file_idx}.jsonl.xz", "wt"
