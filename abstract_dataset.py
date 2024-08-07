"""Types that are used for all datasets."""
from collections.abc import Iterator
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import random
import sys

from tqdm import tqdm

from enums import Jurisdiction
from enums import TaskType
import files
import instruction_manager


@dataclasses.dataclass(frozen=True)
class DataPoint:
    """A data point in the dataset for training an LLM."""
    instruction_language: str
    prompt_language: str
    answer_language: str
    instructions: str
    prompt: str
    answer: str
    task_type: TaskType
    jurisdiction: Jurisdiction
    subset: str


class AbstractDataset:
    """Dataset class that should be subclassed by all datasets.

    This class provides a few helper methods to make it easier to implement a
    new dataset.

    Attributes:
        name: The name of the dataset.
        source: The source of the dataset, e.g. a URL.
        random: A random.Random instance that should be used to make the dataset
            reproducible.
        raw_data_dir: The directory where any raw data is found.
        logger: A logger instance that should be used to log information.
    """

    def __init__(self, name: str, source: str):
        if " " in name:
            raise ValueError("Dataset name should not contain spaces.")
        self.name = name
        self.source = source
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

    def get_data(self, instructions: instruction_manager.InstructionManager) -> Iterator[DataPoint]:
        raise NotImplementedError(
            "This method should yield DataPoint dicts with the following keys: "
            f"{', '.join(DataPoint.__annotations__.keys())}.")

    def build_data_point(self,
                         instruction_language,
                         prompt_language: str,
                         answer_language: str,
                         instructions: str,
                         prompt: str,
                         answer: str,
                         task_type: TaskType = TaskType.UNKNOWN,
                         jurisdiction: Jurisdiction = Jurisdiction.UNKNOWN,
                         subset: str = "") -> DataPoint:
        """Builds a data point.

        Args:
            instruction_language: The language code for the instructions.
            prompt_language: The language of the prompt.
            answer_language: The language of the answer.
            instructions: The text of the instructions.
            prompt: The text of the prompt.
            answer: The text of the answer.
            task_type: The type of the task.
            jurisdiction: The jurisdiction of the task.
            subset: The subset of the dataset the datapoint belongs to.

        Returns:
            A data point with the given attributes.
        """
        del self  # We don't use `self`, but subclasses might.
        return DataPoint(
            instruction_language=instruction_language,
            prompt_language=prompt_language,
            answer_language=answer_language,
            instructions=instructions,
            prompt=prompt,
            answer=answer,
            task_type=task_type,
            jurisdiction=jurisdiction,
            subset=subset,
        )

    def write_json_line(
            self,
            file: files.SupportsWrite,
            datapoint: DataPoint,
    ) -> None:
        """Write a datapoint to a file in JSON format.

        Args:
            file: The file to write to.
            datapoint: The datapoint to write.
        """
        if not datapoint.instructions:
            self.logger.debug(f"datapoint.instruction is empty in {datapoint}")
        if not datapoint.prompt:
            self.logger.debug(f"datapoint.prompt is empty in {datapoint}")
        if not datapoint.answer:
            raise ValueError(
                f"datapoint.answer must not be empty in {datapoint}")
        # text fields are last, so we can easily read the metadata (on servers,
        # for example)
        file.write(
            json.dumps({
                'dataset_name':
                    self.name,
                'subset_name':
                    datapoint.subset,
                'source':
                    self.source,
                'instruction_language':
                    datapoint.instruction_language,
                'prompt_language':
                    datapoint.prompt_language,
                'answer_language':
                    datapoint.answer_language,
                'jurisdiction':
                    datapoint.jurisdiction.name,
                'task_type':
                    datapoint.task_type.name,
                'downloaded_timestamp':
                    datetime.date.today().strftime('%m-%d-%Y'),
                'instruction':
                    datapoint.instructions,
                'prompt':
                    datapoint.prompt,
                'answer':
                    datapoint.answer,
            }) + '\n')

    def _get_output_file_name(self, subset: str = 'MainSubset',
                              split: str = 'train', file_index: int = 0) -> pathlib.Path:
        """Returns the output file name for the given split and index."""
        if not subset:
            subset = 'MainSubset'
        return self.output_dir / f'{self.name}-{subset}-{split}-{file_index}.jsonl.xz'

    def build_instruction_dataset(
            self,
            instructions: instruction_manager.InstructionManager,
            output_dir: os.PathLike = 'data',
            debug_size: int = -1) -> None:
        """Writes a dataset to files.

        We don't want any individual file to get too large, so this method
        writes the dataset to multiple files, each of which is at most
        MAX_FILE_SIZE bytes. This is done by keeping track of the number of
        bytes written to the current file, and opening a new file when the
        current file exceeds MAX_FILE_SIZE. Filenames are handled automatically.

        Args:
            debug_size: If > 0, only write this many datapoints, and log the
              last one for debugging.
        """
        self.logger.info('Building instruction dataset for %s. Loading data...', self.name)

        self.output_dir = pathlib.Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Curry the function to get the file name.
        def get_file_name(file_index):
            return self._get_output_file_name(self.subset, 'train', file_index)

        self.subset = None
        file_index = 0
        writer = None

        for i, datapoint in enumerate(tqdm(self.get_data(instructions))):
            # TODO this solution might not be ideal, because it needs to close and open the filewriter
            #  for every datapoint if we save to different subsets within the same datapoint
            subset = datapoint.subset
            if subset != self.subset: # switch the writer when the subset changes
                if writer is not None:
                    writer.close()
                self.subset = subset
                file_index = 0
                writer = files.SequentialFileWriter(get_file_name)
            if 0 < debug_size <= i:
                self.logger.info('Stopping after %d datapoints.', debug_size)
                self.logger.info('Last datapoint from dataset %s: %s', self.name, datapoint)
                break
            try:
                self.write_json_line(writer, datapoint)
            except ValueError as e:
                self.logger.warning('Skipping datapoint %s due to ValueError: %s', datapoint, e)
            file_index += 1

        if writer is not None:
            writer.close()
