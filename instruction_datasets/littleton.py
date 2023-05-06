import json
import os

import toml

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class Littleton(AbstractDataset):

    def __init__(self):
        super().__init__("Littleton", "https://github.com/grimmelm/littleton")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        json_files = [
            pos_json for pos_json in os.listdir(
                f"{self.raw_data_dir}/littleton/examples/")
            if pos_json.endswith('.json')
        ]
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"

        for json_file in json_files:
            with open(
                    os.path.join(f"{self.raw_data_dir}/littleton/examples/",
                                 json_file), "r") as f:
                loaded_file = json.loads(f.read())[1]
                if isinstance(loaded_file, str):
                    continue
                for example in loaded_file["examples"]:
                    instruction, instruction_language = instructions.sample('littleton_events')
                    text = f"Events: {example['program']}\nAnswer: {example['result']}"
                    prompt = f"Events: {example['program']}"
                    answer = f"Answer: {example['result']}"
                    yield self.build_data_point(instruction_language,
                                                prompt_language, "en",
                                                instruction, prompt, answer,
                                                task_type, jurisdiction)

        json_files = [
            pos_json for pos_json in os.listdir(
                f"{self.raw_data_dir}/littleton/tests/edwards")
            if pos_json.endswith('.toml')
        ]
        for json_file in json_files:
            with open(
                    os.path.join(
                        f"{self.raw_data_dir}/littleton/tests/edwards/",
                        json_file), "r") as f:
                loaded_file = toml.loads(f.read())
                for example in loaded_file["tests"]:
                    if "expected" not in example:
                        continue
                    instruction, instruction_language = instructions.sample('littleton_graph')
                    prompt = f"Events: {example['program']}"
                    answer = f"Answer: {example['expected']}"
                    yield self.build_data_point(instruction_language,
                                                prompt_language, "en",
                                                instruction, prompt, answer,
                                                task_type, jurisdiction)
