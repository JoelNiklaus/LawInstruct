import json
import datetime


def write_json_line(file, text, lang, source):
    file.write(json.dumps({
        "text": text,
        "lang": lang,
        "source": source,
        "downloaded_timestamp": datetime.date.today().strftime("%m-%d-%Y")
    }) + "\n")


def get_output_file_name(category, file_idx=0, split='train'):
    """
    Category should be one of the following: law_instruct, natural_instructions, xp3
    """
    return f"./data/{split}.{category}.{file_idx}.jsonl.xz", "wt"
