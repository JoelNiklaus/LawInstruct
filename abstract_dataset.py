"""Types that are used for all datasets."""
from collections.abc import Iterator
import datetime
import enum
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, TextIO, TypedDict

from tqdm import tqdm

try:
    import lzma as xz
except ImportError:
    import pylzma as xz

MAX_FILE_SIZE = 6.25e8
_FILE_SIZE_CHECK_FREQUENCY = 1000


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
    CODE = enum.auto()
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


class DataPoint(TypedDict):
    """A data point in the dataset for training an LLM."""
    prompt_language: str
    answer_language: str
    text: str
    task_type: TASK_TYPE
    jurisdiction: JURISDICTION
    subset: str


class AbstractDataset:
    """Dataset class that should be subclassed by all datasets.

    This class provides a few helper methods to make it easier to implement a
    new dataset.

    Attributes:
        name: The name of the dataset.
        source: The source of the dataset, e.g. a URL.
        data_dir: The directory where the dataset should be written out.
        random: A random.Random instance that should be used to make the dataset
            reproducible.
        raw_data_dir: The directory where any raw data is found.
        logger: A logger instance that should be used to log information.
    """

    def __init__(self, name: str, source: str, data_dir: os.PathLike = "data"):
        if " " in name:
            raise ValueError("Dataset name should not contain spaces.")
        self.name = name
        self.source = source
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.random: random.Random = random.Random(42)  # make it reproducible

        self.raw_data_dir = "lawinstruct_raw/raw_data"

        self._configure_logging()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _configure_logging() -> None:
        """Configures the logging module to log to stdout."""
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)

    def get_data(self) -> Iterator[DataPoint]:
        raise NotImplementedError(
            "This method should yield DataPoint dicts with the following keys: "
            f"{', '.join(DataPoint.__annotations__.keys())}.")

    def get_instruction_bank(self, language="en"):
        return json.loads(
            Path(f"instruction_banks/{language}.json").read_text())[self.name]

    def build_data_point(self,
                         prompt_language: str,
                         answer_language: str,
                         text: str,
                         task_type: TASK_TYPE = TASK_TYPE.UNKNOWN,
                         jurisdiction: JURISDICTION = JURISDICTION.UNKNOWN,
                         subset: str = "") -> DataPoint:
        """Builds a data point.

        Args:
            prompt_language: The language of the prompt.
            answer_language: The language of the answer.
            text: The text of the prompt and answer.
            task_type: The type of the task.
            jurisdiction: The jurisdiction of the task.
            subset: The subset of the dataset the datapoint belongs to.

        Returns:
            A data point with the given attributes.
        """
        del self  # We don't use `self`, but subclasses might.
        return {
            "prompt_language": prompt_language,
            "answer_language": answer_language,
            "text": text,
            "task_type": task_type,
            "jurisdiction": jurisdiction,
            "subset": subset,
        }

    def write_json_line(self, file: TextIO, datapoint: DataPoint) -> None:
        """Write a datapoint to a file in JSON format.

        Args:
            file: The file to write to.
            datapoint: The datapoint to write.
        """
        if not datapoint['text']:
            raise ValueError(
                f"datapoint['text'] must not be empty in {datapoint}")
        # text is last, so we can easily read the metadata (on servers, for
        # example)
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
                datapoint['text'],
            }) + "\n")

    def get_output_file_name(self,
                             file_idx: int = 0,
                             split: str = 'train') -> str:
        """Builds an output file name.

        Args:
            file_idx: An increasing counter for files of this dataset.
            split: The split of the dataset, e.g. train, dev, test.
        Returns:
            The output file name.
        """
        # we save each dataset to a separate file, so we only need to generate
        # new datasets
        return f"{self.data_dir}/{self.name}.{split}.{file_idx}.jsonl.xz"

    def build_instruction_dataset(self, debug_size: int = -1) -> None:
        """Writes a dataset to files.

        Args:
            debug_size: If > 0, only write this many datapoints, and log the
              last one for debugging.
        """
        output_file_idx = 0
        file = self.open_new_file(output_file_idx)
        self.logger.info(f"Building instruction dataset for {self.name}")
        for i, datapoint in enumerate(tqdm(self.get_data())):
            if i % _FILE_SIZE_CHECK_FREQUENCY == 0:
                if os.path.getsize(self.get_output_file_name(
                        output_file_idx)) > MAX_FILE_SIZE:
                    file.close()
                    output_file_idx += 1
                    file = self.open_new_file(output_file_idx)
            self.write_json_line(file, datapoint)
            if 0 < debug_size <= i:
                self.logger.info(f"Stopping after {debug_size} datapoints")
                self.logger.info(f"Example datapoint: {datapoint}")
                break

        file.close()

    def open_new_file(self, output_file_idx: int) -> TextIO:
        """Opens a new file for writing.

        Args:
            output_file_idx: The index of the file to open.
        Returns:
            The file object.
        """
        filename = self.get_output_file_name(output_file_idx)
        self.logger.info(f"Writing to {filename}")
        return xz.open(filename, "wt")  # do we need append mode here?
