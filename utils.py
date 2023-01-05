import enum
import json
import datetime

MAX_FILE_SIZE = 6.25e8

TASK_TYPE = enum.Enum('TASK_TYPE', [
    # TODO is this detailed enough or do we need to distinguish topic classification from judgment prediction  or NER from argument mining?
    'TEXT_CLASSIFICATION',
    'QUESTION_ANSWERING',
    'SUMMARIZATION',
    'NAMED_ENTITY_RECOGNITION',
    'NATURAL_LANGUAGE_INFERENCE',
    'MULTIPLE_CHOICE',
    'ARGUMENTATION'
    'UNKNOWN'
])


def write_json_line(file, text, lang, source, task_type: TASK_TYPE = TASK_TYPE.UNKNOWN):
    file.write(json.dumps({
        "lang": lang,
        "task_type": task_type,
        "source": source,
        "text": text,
        "downloaded_timestamp": datetime.date.today().strftime("%m-%d-%Y")
    }) + "\n")


def get_output_file_name(dataset_name, file_idx=0, split='train'):
    # we save each dataset to a separate file so we only need to generate new datasets
    return f"./data/{dataset_name}.{split}.{file_idx}.jsonl.xz", "wt"
