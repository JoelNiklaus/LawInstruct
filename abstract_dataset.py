import datetime
import enum
import json
import logging
import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

try:
    import lzma as xz
except ImportError:
    import pylzma as xz

MAX_FILE_SIZE = 6.25e8


class _AutoName(enum.Enum):
    """Enum that overrides `enum.auto` to make value from name instead of from
    next int.

    https://docs.python.org/3.10/library/enum.html#using-automatic-values
    """

    def _generate_next_value_(name: str, start: int, count: int,
                              last_values: list[Any]) -> Any:
        return name


@enum.unique
class TASK_TYPE(_AutoName):
    """The different task types available."""
    # TODO is this detailed enough or do we need to distinguish topic classification from judgment prediction or NER from argument mining?
    TEXT_CLASSIFICATION = enum.auto()
    QUESTION_ANSWERING = enum.auto()
    SUMMARIZATION = enum.auto()
    NAMED_ENTITY_RECOGNITION = enum.auto()
    NATURAL_LANGUAGE_INFERENCE = enum.auto()
    MULTIPLE_CHOICE = enum.auto()
    ARGUMENTATION = enum.auto()
    QUESTION_GENERATION = enum.auto()
    UNKNOWN = enum.auto()


@enum.unique
class JURISDICTION(_AutoName):
    """The jurisdiction where cases are from."""
    # EU
    AUSTRIA = enum.auto()
    BELGIUM = enum.auto()
    BULGARIA = enum.auto()
    CROATIA = enum.auto()
    CZECHIA = enum.auto()
    DENMARK = enum.auto()
    ESTONIA = enum.auto()
    FINLAND = enum.auto()
    FRANCE = enum.auto()
    GERMANY = enum.auto()
    GREECE = enum.auto()
    HUNGARY = enum.auto()
    IRELAND = enum.auto()
    ITALY = enum.auto()
    LATVIA = enum.auto()
    LITHUANIA = enum.auto()
    LUXEMBOURG = enum.auto()
    MALTA = enum.auto()
    NETHERLANDS = enum.auto()
    POLAND = enum.auto()
    PORTUGAL = enum.auto()
    ROMANIA = enum.auto()
    SLOVAKIA = enum.auto()
    SLOVENIA = enum.auto()
    SPAIN = enum.auto()
    SWEDEN = enum.auto()
    # Europa
    EU = enum.auto()
    SWITZERLAND = enum.auto()
    UK = enum.auto()
    # Asia
    CHINA = enum.auto()
    INDIA = enum.auto()
    JAPAN = enum.auto()
    SOUTH_KOREA = enum.auto()
    THAILAND = enum.auto()
    # North America
    US = enum.auto()
    CANADA = enum.auto()
    # South America
    BRAZIL = enum.auto()
    # Other
    INTERNATIONAL = enum.auto()  # international law
    UNKNOWN = enum.auto()  # we don't know the jurisdiction
    N_A = enum.auto()  # Not a legal task


class AbstractDataset:

    def __init__(self, name: str, source: str, data_dir: os.PathLike = "data"):
        self.name = name
        self.source = source
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.random: random.Random = random.Random(42)  # make it reproducible

        self.raw_data_dir = "lawinstruct_raw/raw_data"

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)
        self.logger = logging.getLogger(__name__)

    def get_data(self) -> Iterator[dict]:
        raise NotImplementedError(
            "This method should yield datapoint dicts with the following keys: "
            "prompt_language, answer_language, task_type, jurisdiction, text")

    def get_instruction_bank(self, language="en"):
        return json.loads(
            Path(f"instruction_banks/{language}.json").read_text())[self.name]

    def build_data_point(self,
                         prompt_language: str,
                         answer_language: str,
                         text: str,
                         task_type: TASK_TYPE = TASK_TYPE.UNKNOWN,
                         jurisdiction: JURISDICTION = JURISDICTION.UNKNOWN,
                         subset: str = "") -> dict:
        del self  # We don't use `self`, but subclasses might.
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
        file.write(
            json.dumps({
                "dataset_name":
                self.name,
                "subset_name":
                datapoint.get("subset", ""),
                "source":
                self.source,
                "prompt_language":
                datapoint.get("prompt_language", ""),
                "answer_language":
                datapoint.get("answer_language", ""),
                "jurisdiction":
                datapoint.get("jurisdiction", JURISDICTION.UNKNOWN).name,
                "task_type":
                datapoint.get("task_type", TASK_TYPE.UNKNOWN).name,
                "downloaded_timestamp":
                datetime.date.today().strftime("%m-%d-%Y"),
                "text":
                datapoint[
                    'text'],  # text is last, so we can easily read the metadata on servers for example
            }) + "\n")

    def get_output_file_name(self,
                             file_idx: int = 0,
                             split: str = 'train') -> str:
        # we save each dataset to a separate file, so we only need to generate new datasets
        return f"{self.data_dir}/{self.name}.{split}.{file_idx}.jsonl.xz"

    def build_instruction_dataset(self, debug_size=-1) -> None:
        output_file_idx = 0
        file = self.open_new_file(output_file_idx)
        self.logger.info(f"Building instruction dataset for {self.name}")
        count = 0
        for datapoint in tqdm(self.get_data()):
            if os.path.getsize(self.get_output_file_name(
                    output_file_idx)) > MAX_FILE_SIZE:
                file.close()
                output_file_idx += 1
                file = self.open_new_file(output_file_idx)
            self.write_json_line(file, datapoint)
            count += 1
            if debug_size > 0 and count >= debug_size:
                self.logger.info(f"Stopping after {debug_size} datapoints")
                self.logger.info(f"Example datapoint: {datapoint}")
                break

        file.close()

    def open_new_file(self, output_file_idx: int) -> TextIO:
        filename = self.get_output_file_name(output_file_idx)
        self.logger.info(f"Writing to {filename}")
        return xz.open(filename, "wt")  # do we need append mode here?
