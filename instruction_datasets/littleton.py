import json
import os
import toml
from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class Littleton(AbstractDataset):
    def __init__(self):
        super().__init__("Littleton", "https://github.com/grimmelm/littleton")

    def get_data(self):
        json_files = [pos_json for pos_json in os.listdir("raw_data/littleton/examples/") if pos_json.endswith('.json')]
        instruction_bank = [
            "Consider the law of future interests and conveyances in American property law. Consider the chain of events and then state the interests.",
            "According to American law, consider the chain of events and future interests."]
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.US
        prompt_language = "en"

        for json_file in json_files:
            with open(os.path.join("raw_data/littleton/examples/", json_file), "r") as f:
                loaded_file = json.loads(f.read())[1]
                if isinstance(loaded_file, str):
                    continue
                for example in loaded_file["examples"]:
                    text = f"{self.random.choice(instruction_bank)}\n\nEvents: {example['program']}\nAnswer: {example['result']}"
                    yield self.build_data_point(prompt_language, "en", text, task_type, jurisdiction)

        json_files = [pos_json for pos_json in os.listdir("raw_data/littleton/tests/edwards") if
                      pos_json.endswith('.toml')]
        instruction_bank = [
            "Consider the law of future interests and conveyances in American property law. Consider the chain of events and then output a graph structure representing the events.",
            "According to American law, consider the chain of events and future interests. Output a graph structure representing the events and any interests."]
        for json_file in json_files:
            with open(os.path.join("raw_data/littleton/tests/edwards/", json_file), "r") as f:
                loaded_file = toml.loads(f.read())
                for example in loaded_file["tests"]:
                    if "expected" not in example:
                        continue
                    text = f"{self.random.choice(instruction_bank)}\n\nEvents: {example['program']}\nAnswer: {example['expected']}"
                    yield self.build_data_point(prompt_language, "en", text, task_type, jurisdiction)
