import datetime
import enum
import json
import logging
import os
import random
from collections.abc import Iterator
from typing import TextIO

from tqdm import tqdm

try:
    import lzma as xz
except ImportError:
    import pylzma as xz

MAX_FILE_SIZE = 6.25e8

TASK_TYPE = enum.Enum('TASK_TYPE', [
    # TODO is this detailed enough or do we need to distinguish topic classification from judgment prediction or NER from argument mining?
    'TEXT_CLASSIFICATION',
    'QUESTION_ANSWERING',
    'SUMMARIZATION',
    'NAMED_ENTITY_RECOGNITION',
    'NATURAL_LANGUAGE_INFERENCE',
    'MULTIPLE_CHOICE',
    'ARGUMENTATION',
    'ANSWER_GENERATION',
    'QUESTION_GENERATION',
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


class AbstractDataset:
    def __init__(self, name: str, source: str, data_dir: os.PathLike = "data"):
        self.name = name
        self.source = source
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.random: random.Random = random.Random(42)  # make it reproducible
        self.logger = logging.getLogger(__name__)

    def get_data(self) -> Iterator[dict]:
        raise NotImplementedError("This method should yield datapoint dicts with the following keys: "
                                  "prompt_language, answer_language, task_type, jurisdiction, text")

    def build_data_point(self,
                         prompt_language: str,
                         answer_language: str,
                         text: str,
                         task_type: TASK_TYPE = TASK_TYPE.UNKNOWN,
                         jurisdiction: JURISDICTION = JURISDICTION.UNKNOWN,
                         subset: str = None) -> dict:
        return {
            "prompt_language": prompt_language,
            "answer_language": answer_language,
            "text": text,
            "task_type": task_type,
            "jurisdiction": jurisdiction,
            "subset": subset,
        }

    def write_json_line(self, file: TextIO, datapoint: dict) -> None:
        assert datapoint['text'], "datapoint['text'] must not be empty"
        file.write(json.dumps({
            "dataset_name": self.name,
            "subset_name": datapoint.get("subset", None),
            "source": self.source,
            "prompt_language": datapoint.get("prompt_language", None),
            "answer_language": datapoint.get("answer_language", None),
            "jurisdiction": datapoint.get("jurisdiction", JURISDICTION.UNKNOWN).name,
            "task_type": datapoint.get("task_type", TASK_TYPE.UNKNOWN).name,
            "downloaded_timestamp": datetime.date.today().strftime("%m-%d-%Y"),
            "text": datapoint['text'],  # text is last so we can easily read the metadata on servers for example
        }) + "\n")

    def get_output_file_name(self, file_idx: int = 0, split: str = 'train') -> str:
        # we save each dataset to a separate file, so we only need to generate new datasets
        return f"{self.data_dir}/{self.name}.{split}.{file_idx}.jsonl.xz"

    def build_instruction_dataset(self) -> None:
        output_file_idx = 0
        file = self.open_new_file(output_file_idx)
        for datapoint in tqdm(self.get_data()):
            if os.path.getsize(self.get_output_file_name(output_file_idx)) > MAX_FILE_SIZE:
                file.close()
                output_file_idx += 1
                file = self.open_new_file(output_file_idx)
            self.write_json_line(file, datapoint)
        file.close()

    def open_new_file(self, output_file_idx: int) -> TextIO:
        filename = self.get_output_file_name(output_file_idx)
        self.logger.info(f"Writing to {filename}")
        return xz.open(filename, "wt")  # do we need append mode here?
