import json
import os

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class Lila(AbstractDataset):

    def __init__(self):
        super().__init__("Lila", "https://github.com/allenai/Lila")

    def get_data(self):
        json_files = [
            pos_json
            for pos_json in os.listdir(f"{self.raw_data_dir}/all_lila/")
            if pos_json.endswith('.json')
        ]
        instruction_bank = [
            "Consider the following question. Write a Python program to solve it.",
            "Write a Python program to solve the following question, denote it as \"Program:\". Provide the output as \"Answer:\"."
        ]
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.N_A
        instruction_language = "en"
        prompt_language = "en"

        for json_file in json_files:
            with open(os.path.join(f"{self.raw_data_dir}/all_lila/", json_file),
                      "r") as f:
                loaded_file = json.loads(f.read())
                for example in loaded_file["Instances"]:
                    if example["split"] != "train":
                        continue
                    for program, answer in zip(example['Output Program'],
                                               example['Output Answer']):
                        instruction = self.random.choice(instruction_bank)
                        prompt = f"Question: {example['Input']}\n" \
                                    f"Program:\n```python\n{program}\n```\n"
                        answer = f"Answer: {answer}"
                        yield self.build_data_point(instruction_language, prompt_language, "en",
                                                    instruction, prompt, answer,
                                                    task_type, jurisdiction)
